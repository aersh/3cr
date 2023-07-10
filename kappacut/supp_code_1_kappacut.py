'''
This code was used as a basis for all the Kappa simulations in Ershova et al. "Enzyme-free exponential amplification via growth
and scision of crisscross ribbons from single-stranded DNA components". It is provided as-is to exemplify the implementation of
the three-point constraint for crisscross rule generation. Further analysis and simulation scripts are available upon request.
'''

from KappaUtilities import kappathings as kt
from KappaUtilities import kappasnap as ks
from KappaUtilities import kappaviz as viz
from KappaUtilities import kappagraph as kg
from KappaUtilities import kappamorph as km
from KappaUtilities import kapplot as kp

import numpy as np
import subprocess
import time
import os
from multiprocessing import Process

def codegen(directory,fname,n,t,inter_k_on,inter_k_off,intra_k_on,intra_k_off,
            inter_wobble_k_on,inter_wobble_k_off,intra_wobble_k_on,intra_wobble_k_off,
            cslat_count,allow_growth_rebind,wobble_everywhere,snap_interval,maxConsecutiveClash,n_cx=None,prebound_cslats=False):

    # can define any array of shape (*,n) for x slats – this is the free parameter in design here, * should be even
    # elements in the array are a unique word identifier, and are used for matching up x- and y- words
    xa = np.arange(1,1+n*(n+t+t)).reshape(n,n+t+t) # with toeholds

    # generates matrices defining relationship between x- and y- slats
    # throughout this script, t1,t2,s1,s2,n1,n2,n3 are "helper" matrices for parts of the transformations
    # needed to get the matrices for x,y,seed,nucy

    t1 = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(xa)]) # flip even rows (3'-->5')
    t2 = np.transpose([np.roll(arr,n-1-i) for i,arr in enumerate(np.transpose(np.flipud(t1)))]) # crisscross pattern
    if t%2 == 0:
        ya = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(t2)])[::-1] # flip even rows (5'-->3')
    else:
        ya = np.array([row[::-1] if i%2 == 1 else row for i,row in enumerate(t2)])[::-1] # flip odd rows (5'-->3')
    ya = np.roll(ya,-t,axis=0) # shift around to correspond to normal ordering

    xfull = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(xa)]) # flip to allow directional rule generation
    yfull = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(ya)]) # flip to allow directional rule generation

    x = np.array([row[:-t] if t != 0 else row for row in xfull]) # full slat sequence
    y = np.array([row[:-t] if t != 0 else row for row in yfull]) # full slat sequence

    xcore = np.array([row[t:-t] if t != 0 else row for row in xfull]) # no extensions
    ycore = np.array([row[t:-t] if t != 0 else row for row in yfull]) # no extensions

    if n_cx == None:
        n_cx = int(n/2) # specify number of cut-x slats used (defaults to slight cut-x bias)
    n_cy = n - 1 - n_cx
    print('n_cx:',n_cx,'n_cy:',n_cy)

    cx = [row[t+n-n_cx+i:] for i,row in enumerate(xfull[::-1]) if i < n_cx][::-1]

    cy = [row[t+n-n_cy+i:] for i,row in enumerate(yfull[::-1][n-n_cy-1:]) if i < n_cy][::-1]

    # standard wobble arrangement used in experiments
    wobble_sites = []
    wobnx = n_cx - 1
    wobny = n_cy - 1
    for slat in cx[-wobnx:]+cy[-wobny:]:
        wobble_sites.append(slat[-(t+1)])
    
    if wobble_everywhere == True:
        wobble_sites = []
        for slat in cx+cy:
            for i in range(1,1+len(slat)-t):
                wobble_sites.append(slat[-(t+i)])
    
    print('Wobble sites:', wobble_sites)


    rules = []

    # Generate rules for x and y interactions

    def rule_str(i,j,k,l): # create Kappa rule
        yi = int(np.where(y==i)[0])
        yj = int(np.where(y==j)[0])
        assert yi == yj
        xj = int(np.where(x==j)[0])
        xk = int(np.where(x==k)[0])
        assert xj == xk
        yk = int(np.where(y==k)[0])
        yl = int(np.where(y==l)[0])
        assert yk == yl
        xi = int(np.where(x==i)[0])
        xl = int(np.where(x==l)[0])
        assert xi == xl
        if i in wobble_sites:
            return [f"Y{yi}(y{i}[.], y{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), X{xi}(x{i}[.], x{l}[4]) <-> Y{yi}(y{i}[1], y{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), X{xi}(x{i}[1], x{l}[4]) @ 'intra_wobble_k_on', 'intra_wobble_k_off'",
                    f"Y{yi}(y{i}[1]), X{xi}(x{i}[1]) -> Y{yi}(y{i}[.]), X{xi}(x{i}[.]) @ 'intra_wobble_k_off'"]
        else:
            return [f"Y{yi}(y{i}[.], y{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), X{xi}(x{i}[.], x{l}[4]) <-> Y{yi}(y{i}[1], y{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), X{xi}(x{i}[1], x{l}[4]) @ 'intra_k_on', 'intra_k_off'",
                    f"Y{yi}(y{i}[1]), X{xi}(x{i}[1]) -> Y{yi}(y{i}[.]), X{xi}(x{i}[.]) @ 'intra_k_off'"]

    core_sites = list(set(xcore.flatten()) & set(ycore.flatten())) # binding sites in core and not extensions
    assert set(xcore.flatten()) == set(ycore.flatten())

    # 4 possible traversals to generate intramolecular rules

    # y_left --> x_left --> y_right
    for i in core_sites:
        y_a = ycore[(ycore==i).any(axis=1)].flatten()
        for step_a in range(1,n):
            index_a = int(np.argwhere(y_a==i)) + step_a
            if index_a < n:
                j = y_a[index_a]
                x_a = xcore[(xcore==j).any(axis=1)].flatten()
                for step_b in range(1,n):
                    index_b = int(np.argwhere(x_a==j)) + step_b
                    if index_b < n:
                        k = x_a[index_b]
                        y_b = ycore[(ycore==k).any(axis=1)].flatten()
                        index_c = int(np.argwhere(y_b==k)) - step_a
                        if index_c >= 0:
                            l = y_b[index_c]
                            rules += rule_str(i,j,k,l)
                            
    # y_left --> x_right --> y_right
    for i in core_sites:
        y_a = ycore[(ycore==i).any(axis=1)].flatten()
        for step_a in range(1,n):
            index_a = int(np.argwhere(y_a==i)) + step_a
            if index_a < n:
                j = y_a[index_a]
                x_a = xcore[(xcore==j).any(axis=1)].flatten()
                for step_b in range(1,n):
                    index_b = int(np.argwhere(x_a==j)) - step_b
                    if index_b >= 0:
                        k = x_a[index_b]
                        y_b = ycore[(ycore==k).any(axis=1)].flatten()
                        index_c = int(np.argwhere(y_b==k)) - step_a
                        if index_c >= 0:
                            l = y_b[index_c]
                            rules += rule_str(i,j,k,l)
                            
    # y_right --> x_left --> y_left
    for i in core_sites:
        y_a = ycore[(ycore==i).any(axis=1)].flatten()
        for step_a in range(1,n):
            index_a = int(np.argwhere(y_a==i)) - step_a
            if index_a >= 0:
                j = y_a[index_a]
                x_a = xcore[(xcore==j).any(axis=1)].flatten()
                for step_b in range(1,n):
                    index_b = int(np.argwhere(x_a==j)) + step_b
                    if index_b < n:
                        k = x_a[index_b]
                        y_b = ycore[(ycore==k).any(axis=1)].flatten()
                        index_c = int(np.argwhere(y_b==k)) + step_a
                        if index_c < n:
                            l = y_b[index_c]
                            rules += rule_str(i,j,k,l)                        
                                                    
    # y_right --> x_right --> y_left
    for i in core_sites:
        y_a = ycore[(ycore==i).any(axis=1)].flatten()
        for step_a in range(1,n):
            index_a = int(np.argwhere(y_a==i)) - step_a
            if index_a >= 0:
                j = y_a[index_a]
                x_a = xcore[(xcore==j).any(axis=1)].flatten()
                for step_b in range(1,n):
                    index_b = int(np.argwhere(x_a==j)) - step_b
                    if index_b >= 0:
                        k = x_a[index_b]
                        y_b = ycore[(ycore==k).any(axis=1)].flatten()
                        index_c = int(np.argwhere(y_b==k)) + step_a
                        if index_c < n:
                            l = y_b[index_c]
                            rules += rule_str(i,j,k,l)

    # cx bridged by core x

    def rule_str_cx_core(i,j,k,l): # create Kappa rule
        yi = int(np.where(y==i)[0])
        yj = int(np.where(y==j)[0])
        assert yi == yj
        xj = int(np.where(x==j)[0])
        xk = int(np.where(x==k)[0])
        assert xj == xk
        yk = int(np.where(y==k)[0])
        yl = int(np.where(y==l)[0])
        assert yk == yl
        for cnum,cslat in enumerate(cx):
            if i in cslat:
                xi = cnum
            if l in cslat:
                xl = cnum
        assert xi == xl
        return [f"Y{yi}(y{i}[.], y{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), CX{xi}(cx{i}[.], cx{l}[4]) <-> Y{yi}(y{i}[1], y{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), CX{xi}(cx{i}[1], cx{l}[4]) @ 'intra_k_on', 'intra_k_off'",
                f"Y{yi}(y{i}[1]), CX{xi}(cx{i}[1]) -> Y{yi}(y{i}[.]), CX{xi}(cx{i}[.]) @ 'intra_k_off'"]

    # y_left --> x_left --> y_right
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,n+t):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < n+t:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()
                    if len(x_a) != 0: # test that there's actually an x-slat identified, not e.g. just extension    
                        for step_b in range(1,len(slat)):
                            index_b = int(np.argwhere(x_a==j)) + step_b
                            if index_b < n+t:
                                k = x_a[index_b]
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) - step_a
                                if index_c >= 0:
                                    l = y_b[index_c]
                                    rules += rule_str_cx_core(i,j,k,l)
                                    
    # y_left --> x_right --> y_right
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,n+t):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < n+t:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()
                    if len(x_a) != 0:   
                        for step_b in range(1,n+t):
                            index_b = int(np.argwhere(x_a==j)) - step_b
                            if index_b >= 0:
                                k = x_a[index_b]
                                if k in core_sites:
                                    y_b = y[(y==k).any(axis=1)].flatten()
                                    index_c = int(np.argwhere(y_b==k)) - step_a
                                    if index_c >= 0:
                                        l = y_b[index_c]
                                        if l in slat:
                                            rules += rule_str_cx_core(i,j,k,l)
                                            
    # y_right --> x_left --> y_left
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,n+t):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()
                    if len(x_a) != 0:   
                        for step_b in range(1,len(slat)):
                            index_b = int(np.argwhere(x_a==j)) + step_b
                            if index_b < n+t:
                                k = x_a[index_b]
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) + step_a
                                if index_c < n+t:
                                    l = y_b[index_c]
                                    rules += rule_str_cx_core(i,j,k,l)
                                    
    # y_right --> x_right --> y_left
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,n+t):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()
                    if len(x_a) != 0: 
                        for step_b in range(1,n+t):
                            index_b = int(np.argwhere(x_a==j)) - step_b
                            if index_b >= 0:
                                k = x_a[index_b]
                                if k in core_sites:
                                    y_b = y[(y==k).any(axis=1)].flatten()
                                    index_c = int(np.argwhere(y_b==k)) + step_a
                                    if index_c < n+t:
                                        l = y_b[index_c]
                                        if l in slat:
                                            rules += rule_str_cx_core(i,j,k,l)

    # cx bridged by cx

    def rule_str_cx_cx(i,j,k,l): # create Kappa rule
        yi = int(np.where(y==i)[0])
        yj = int(np.where(y==j)[0])
        assert yi == yj
        for cnum,cslat in enumerate(cx):
            if j in cslat:
                xj = cnum
            if k in cslat:
                xk = cnum
        assert xj == xk
        yk = int(np.where(y==k)[0])
        yl = int(np.where(y==l)[0])
        assert yk == yl
        for cnum,cslat in enumerate(cx):
            if i in cslat:
                xi = cnum
            if l in cslat:
                xl = cnum
        assert xi == xl
        return [f"Y{yi}(y{i}[.], y{j}[2]), CX{xj}(cx{j}[2], cx{k}[3]), Y{yk}(y{k}[3], y{l}[4]), CX{xi}(cx{i}[.], cx{l}[4]) <-> Y{yi}(y{i}[1], y{j}[2]), CX{xj}(cx{j}[2], cx{k}[3]), Y{yk}(y{k}[3], y{l}[4]), CX{xi}(cx{i}[1], cx{l}[4]) @ 'intra_k_on', 'intra_k_off'",
                f"Y{yi}(y{i}[1]), CX{xi}(cx{i}[1]) -> Y{yi}(y{i}[.]), CX{xi}(cx{i}[.]) @ 'intra_k_off'"]

    # y_left --> x_left --> y_right
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,len(cx)):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < n+t:
                    j = y_a[index_a]
                    if j in np.concatenate(cx): # check this is bridgeable by cx slat and not just x-slat
                        for cnum,cslat in enumerate(cx):
                            if j in cslat:
                                x_a = cslat  
                        for step_b in range(1,len(slat)):
                            index_b = int(np.argwhere(x_a==j)) + step_b
                            if index_b < len(x_a):
                                k = x_a[index_b]
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) - step_a
                                if index_c >= 0:
                                    l = y_b[index_c]
                                    rules += rule_str_cx_cx(i,j,k,l)
                                        
    # y_left --> x_right --> y_right
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,len(cx)):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < n+t:
                    j = y_a[index_a]
                    if j in np.concatenate(cx): # check this is bridgeable by cx slat and not just x-slat
                        for cnum,cslat in enumerate(cx):
                            if j in cslat:
                                x_a = cslat
                        for step_b in range(1,len(slat)):
                            index_b = int(np.argwhere(x_a==j)) - step_b
                            if index_b >= 0:
                                k = x_a[index_b]
                                if k in core_sites:
                                    y_b = y[(y==k).any(axis=1)].flatten()
                                    index_c = int(np.argwhere(y_b==k)) - step_a
                                    if index_c >= 0:
                                        l = y_b[index_c]
                                        if l in slat:
                                            rules += rule_str_cx_cx(i,j,k,l)
                                                
    # y_right --> x_left --> y_left
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,len(cx)):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    if j in np.concatenate(cx): # check this is bridgeable by cx slat and not just x-slat
                        for cnum,cslat in enumerate(cx):
                            if j in cslat:
                                x_a = cslat   
                        for step_b in range(1,len(slat)):
                            index_b = int(np.argwhere(x_a==j)) + step_b
                            if index_b < len(x_a):
                                k = x_a[index_b]
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) + step_a
                                if index_c < n+t:
                                    l = y_b[index_c]
                                    rules += rule_str_cx_cx(i,j,k,l)
                                    
    # y_right --> x_right --> y_left
    for slat in cx:
        for i in slat:
            y_a = y[(y==i).any(axis=1)].flatten()
            for step_a in range(1,len(cx)):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    if j in np.concatenate(cx): # check this is bridgeable by cx slat and not just x-slat
                        for cnum,cslat in enumerate(cx):
                            if j in cslat:
                                x_a = cslat   
                        for step_b in range(1,len(slat)):
                            index_b = int(np.argwhere(x_a==j)) - step_b
                            if index_b >= 0:
                                k = x_a[index_b]
                                if k in core_sites:
                                    y_b = y[(y==k).any(axis=1)].flatten()
                                    index_c = int(np.argwhere(y_b==k)) + step_a
                                    if index_c < n+t:
                                        l = y_b[index_c]
                                        if l in slat:
                                            rules += rule_str_cx_cx(i,j,k,l)

    # cy bridged by core y

    def rule_str_cy_core(i,j,k,l): # create Kappa rule
        for cnum,cslat in enumerate(cy):
            if i in cslat:
                yi = cnum
            if j in cslat:
                yj = cnum
        assert yi == yj
        xj = int(np.where(x==j)[0])
        xk = int(np.where(x==k)[0])
        assert xj == xk
        yk = int(np.where(y==k)[0])
        yl = int(np.where(y==l)[0])
        assert yk == yl
        xi = int(np.where(x==i)[0])
        xl = int(np.where(x==l)[0])
        assert xi == xl
        return [f"CY{yi}(cy{i}[.], cy{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), X{xi}(x{i}[.], x{l}[4]) <-> CY{yi}(cy{i}[1], cy{j}[2]), X{xj}(x{j}[2], x{k}[3]), Y{yk}(y{k}[3], y{l}[4]), X{xi}(x{i}[1], x{l}[4]) @ 'intra_k_on', 'intra_k_off'",
                f"CY{yi}(cy{i}[1]), X{xi}(x{i}[1]) -> CY{yi}(cy{i}[.]), X{xi}(x{i}[.]) @ 'intra_k_off'"]

    # y_left --> x_left --> y_right
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < len(y_a):
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()    
                    for step_b in range(1,n+t):
                        index_b = int(np.argwhere(x_a==j)) + step_b
                        if index_b < n+t:
                            k = x_a[index_b]
                            if k in core_sites:
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) - step_a
                                if index_c >= 0:
                                    l = y_b[index_c]
                                    if l in core_sites: # make sure it's bridgeable by a core y-slat
                                        rules += rule_str_cy_core(i,j,k,l)
                                        
    # y_left --> x_right --> y_right
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < len(y_a):
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten() 
                    for step_b in range(1,n+t):
                        index_b = int(np.argwhere(x_a==j)) - step_b
                        if index_b >= 0:
                            k = x_a[index_b]
                            if k in core_sites:
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) - step_a
                                if index_c >= 0:
                                    l = y_b[index_c]
                                    rules += rule_str_cy_core(i,j,k,l)
                                    
    # y_right --> x_left --> y_left
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten() 
                    for step_b in range(1,n+t):
                        index_b = int(np.argwhere(x_a==j)) + step_b
                        if index_b < n+t:
                            k = x_a[index_b]
                            if k in core_sites:
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) + step_a
                                if index_c < n+t:
                                    l = y_b[index_c]
                                    rules += rule_str_cy_core(i,j,k,l)
                                    
    # y_right --> x_right --> y_left
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()
                    for step_b in range(1,n+t):
                        index_b = int(np.argwhere(x_a==j)) - step_b
                        if index_b >= 0:
                            k = x_a[index_b]
                            if k in core_sites:
                                y_b = y[(y==k).any(axis=1)].flatten()
                                index_c = int(np.argwhere(y_b==k)) + step_a
                                if index_c < n+t:
                                    l = y_b[index_c]
                                    rules += rule_str_cy_core(i,j,k,l)

    # cy bridged by cy

    def rule_str_cy_cy(i,j,k,l): # create Kappa rule
        for cnum,cslat in enumerate(cy):
            if i in cslat:
                yi = cnum
            if j in cslat:
                yj = cnum
        assert yi == yj
        xj = int(np.where(x==j)[0])
        xk = int(np.where(x==k)[0])
        assert xj == xk
        for cnum,cslat in enumerate(cy):
            if k in cslat:
                yk = cnum
            if l in cslat:
                yl = cnum
        assert yk == yl
        xi = int(np.where(x==i)[0])
        xl = int(np.where(x==l)[0])
        assert xi == xl
        return [f"CY{yi}(cy{i}[.], cy{j}[2]), X{xj}(x{j}[2], x{k}[3]), CY{yk}(cy{k}[3], cy{l}[4]), X{xi}(x{i}[.], x{l}[4]) <-> CY{yi}(cy{i}[1], cy{j}[2]), X{xj}(x{j}[2], x{k}[3]), CY{yk}(cy{k}[3], cy{l}[4]), X{xi}(x{i}[1], x{l}[4]) @ 'intra_k_on', 'intra_k_off'",
                f"CY{yi}(cy{i}[1]), X{xi}(x{i}[1]) -> CY{yi}(cy{i}[.]), X{xi}(x{i}[.]) @ 'intra_k_off'"]

    # y_left --> x_left --> y_right
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < len(y_a):
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()    
                    for step_b in range(1,len(cy)):
                        index_b = int(np.argwhere(x_a==j)) + step_b
                        if index_b < len(x_a):
                            k = x_a[index_b]
                            if k in np.concatenate(cy): # check this is bridgeable by cy slat not just y-slat
                                for cnum,cslat in enumerate(cy):
                                    if k in cslat:
                                        y_b = cslat
                                index_c = int(np.argwhere(y_b==k)) - step_a
                                if index_c >= 0:
                                    l = y_b[index_c]
                                    rules += rule_str_cy_cy(i,j,k,l)
                                    
    # y_left --> x_right --> y_right
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) + step_a
                if index_a < len(y_a):
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten() 
                    for step_b in range(1,len(cy)):
                        index_b = int(np.argwhere(x_a==j)) - step_b
                        if index_b >= 0:
                            k = x_a[index_b]
                            if k in np.concatenate(cy):
                                for cnum,cslat in enumerate(cy):
                                    if k in cslat:
                                        y_b = cslat
                                index_c = int(np.argwhere(y_b==k)) - step_a
                                if index_c >= 0:
                                    l = y_b[index_c]
                                    rules += rule_str_cy_cy(i,j,k,l)
                                    
    # y_right --> x_left --> y_left
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten() 
                    for step_b in range(1,len(cy)):
                        index_b = int(np.argwhere(x_a==j)) + step_b
                        if index_b < n+t:
                            k = x_a[index_b]
                            if k in np.concatenate(cy):
                                for cnum,cslat in enumerate(cy):
                                    if k in cslat:
                                        y_b = cslat
                                index_c = int(np.argwhere(y_b==k)) + step_a
                                if index_c < n+t:
                                    l = y_b[index_c]
                                    rules += rule_str_cy_cy(i,j,k,l)
                                    
    # y_right --> x_right --> y_left
    for slat in cy:
        for i in slat:
            for cnum,cslat in enumerate(cy):
                if i in cslat:
                    y_a = cslat
            for step_a in range(1,len(slat)):
                index_a = int(np.argwhere(y_a==i)) - step_a
                if index_a >= 0:
                    j = y_a[index_a]
                    x_a = x[(x==j).any(axis=1)].flatten()
                    for step_b in range(1,len(cy)):
                        index_b = int(np.argwhere(x_a==j)) - step_b
                        if index_b >= 0:
                            k = x_a[index_b]
                            if k in np.concatenate(cy):
                                for cnum,cslat in enumerate(cy):
                                    if k in cslat:
                                        y_b = cslat
                                index_c = int(np.argwhere(y_b==k)) + step_a
                                if index_c < len(y_b):
                                    l = y_b[index_c]
                                    rules += rule_str_cy_cy(i,j,k,l)

    # bimolecular rules

    # binding of free x to y in existing complex and free y to x in existing complex

    # exclude these to prevent ribbon from growing

    if allow_growth_rebind == True:

        for i,s in enumerate(x):
            for j in range(len(s)):
                if s[j] in core_sites:
                    slat = [f'x{s[k]}[.]' for k in range(len(s))]
                    l = int(np.where(y==s[j])[0])
                    lhs = f"X{i}({', '.join(slat)}), Y{l}(y{s[j]}[.])"
                    slat[j] = f'x{s[j]}[1]'
                    rhs = f"X{i}({', '.join(slat)}), Y{l}(y{s[j]}[1])"
                    if i in wobble_sites:
                        rule = lhs + ' <-> ' + rhs + " @ 'inter_wobble_k_on', 'inter_wobble_k_off'"
                    else:
                        rule = lhs + ' <-> ' + rhs + " @ 'inter_k_on', 'inter_k_off'"
                    rules.append(rule)

        for i,s in enumerate(y):
            for j in range(len(s)):
                if s[j] in core_sites:
                    slat = [f'y{s[k]}[.]' for k in range(len(s))]
                    l = int(np.where(x==s[j])[0])
                    lhs = f"Y{i}({', '.join(slat)}), X{l}(x{s[j]}[.])"
                    slat[j] = f'y{s[j]}[1]'
                    rhs = f"Y{i}({', '.join(slat)}), X{l}(x{s[j]}[1])"
                    if i in wobble_sites:
                        rule = lhs + ' <-> ' + rhs + " @ 'inter_wobble_k_on', 'inter_wobble_k_off'"
                    else:
                        rule = lhs + ' <-> ' + rhs + " @ 'inter_k_on', 'inter_k_off'"
                    rules.append(rule)
                    
    if prebound_cslats != True or allow_growth_rebind == True:

        # binding of free cx to y in existing complex and free cy to x in existing complex

        for i,s in enumerate(cx):
            for j in range(len(s)):
                slat = [f'cx{s[k]}[.]' for k in range(len(s))]
                l = int(np.where(y==s[j])[0])
                lhs = f"CX{i}({', '.join(slat)}), Y{l}(y{s[j]}[.])"
                slat[j] = f'cx{s[j]}[1]'
                rhs = f"CX{i}({', '.join(slat)}), Y{l}(y{s[j]}[1])"
                rule = lhs + ' <-> ' + rhs + " @ 'inter_k_on', 'inter_k_off'"
                rules.append(rule)

        for i,s in enumerate(cy):
            for j in range(len(s)):
                slat = [f'cy{s[k]}[.]' for k in range(len(s))]
                l = int(np.where(x==s[j])[0])
                lhs = f"CY{i}({', '.join(slat)}), X{l}(x{s[j]}[.])"
                slat[j] = f'cy{s[j]}[1]'
                rhs = f"CY{i}({', '.join(slat)}), X{l}(x{s[j]}[1])"
                rule = lhs + ' <-> ' + rhs + " @ 'inter_k_on', 'inter_k_off'"
                rules.append(rule)

        # binding of free x to cy in existing complex and free y to cx in existing complex

        for i,s in enumerate(x):
            for j in range(len(s)):
                if s[j] in np.concatenate(cy):
                    slat = [f'x{s[k]}[.]' for k in range(len(s))]
                    for cnum,cslat in enumerate(cy):
                        if s[j] in cslat:
                            l = cnum
                    lhs = f"X{i}({', '.join(slat)}), CY{l}(cy{s[j]}[.])"
                    slat[j] = f'x{s[j]}[1]'
                    rhs = f"X{i}({', '.join(slat)}), CY{l}(cy{s[j]}[1])"
                    rule = lhs + ' <-> ' + rhs + " @ 'inter_k_on', 'inter_k_off'"
                    rules.append(rule)

        for i,s in enumerate(y):
            for j in range(len(s)):
                if s[j] in np.concatenate(cx):
                    slat = [f'y{s[k]}[.]' for k in range(len(s))]
                    for cnum,cslat in enumerate(cx):
                        if s[j] in cslat:
                            l = cnum
                    lhs = f"Y{i}({', '.join(slat)}), CX{l}(cx{s[j]}[.])"
                    slat[j] = f'y{s[j]}[1]'
                    rhs = f"Y{i}({', '.join(slat)}), CX{l}(cx{s[j]}[1])"
                    rule = lhs + ' <-> ' + rhs + " @ 'inter_k_on', 'inter_k_off'"
                    rules.append(rule)

    agents = []

    for i,s in enumerate(x):
        agents.append(f"%agent: X{i}({', '.join([f'x{s[j]}' for j in range(len(s))])})")
    for i,s in enumerate(y):
        agents.append(f"%agent: Y{i}({', '.join([f'y{s[j]}' for j in range(len(s))])})")
    for i,s in enumerate(cx):
        agents.append(f"%agent: CX{i}({', '.join([f'cx{s[j]}' for j in range(len(s))])})")
    for i,s in enumerate(cy):
        agents.append(f"%agent: CY{i}({', '.join([f'cy{s[j]}' for j in range(len(s))])})")

    # generate helper arrays that are big to make longer ribbon

    # can define any array of shape (*,n) for x slats – this is the free parameter in design here, * should be even
    # elements in the array are a unique word identifier, and are used for matching up x- and y- words
    hxa = np.arange(1,1+2*n*(n+t+t)).reshape(2*n,n+t+t) # with toeholds

    # generates matrices defining relationship between x- and y- slats
    # throughout this script, t1,t2,s1,s2,n1,n2,n3 are "helper" matrices for parts of the transformations
    # needed to get the matrices for x,y,seed,nucy

    ht1 = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(hxa)]) # flip even rows (3'-->5')
    ht2 = np.transpose([np.roll(arr,n-1-i) for i,arr in enumerate(np.transpose(np.flipud(ht1)))]) # crisscross pattern
    if t%2 == 0:
        hya = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(ht2)])[::-1] # flip even rows (5'-->3')
    else:
        hya = np.array([row[::-1] if i%2 == 1 else row for i,row in enumerate(ht2)])[::-1] # flip odd rows (5'-->3')
    hya = np.roll(hya,-t,axis=0) # shift around to correspond to normal ordering

    hxfull = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(hxa)]) # flip to allow directional rule generation
    hyfull = np.array([row[::-1] if i%2 == 0 else row for i,row in enumerate(hya)]) # flip to allow directional rule generation

    hx = np.array([row[:-t] if t != 0 else row for row in hxfull]) # full slat sequence
    hy = np.array([row[:-t] if t != 0 else row for row in hyfull]) # full slat sequence
    
    if prebound_cslats == True:
        
        hxl = hx.tolist()
        for i,s in enumerate(hxl[:int(n/2)-1]):
            hxl[i] = ['.' if j < int(n/2)-1-i+t and j>=t else b for j,b in enumerate(s)]

        for i,s in enumerate(hxl[-int(n/2):]):
            hxl[hxl.index(s)] = [b if j < n-1-i+t else '.' for j,b in enumerate(s)]

        for i,s in enumerate(hxl):
            hxl[i] = ['.' if j < t and b not in [cb+(n*(n+t+t)) for cs in cy for cb in cs] else b for j,b in enumerate(s)]


        hyl = hy.tolist()
        for i,s in enumerate(hyl[n+1:n+int(n/2)]):
            hyl[hyl.index(s)] = [b if j < n-1-i+t else '.' for j,b in enumerate(s)]

        for i,s in enumerate(hyl[-int(n/2):]):
            hyl[hyl.index(s)] = ['.' if j < int(n/2)-i+t and j>=t else b for j,b in enumerate(s)]

        for i,s in enumerate(hyl):
            hyl[i] = ['.' if j < t and b not in [cb for cs in cx for cb in cs] else b for j,b in enumerate(s)]

        ribbon = []

        for i,s in enumerate(hxl):
            ribbon.append(f"X{i%n}({', '.join([f'x{x[i%n][j]}[{bs}]' for j,bs in enumerate(s)])})")

        for i,s in enumerate(hyl):
            ribbon.append(f"Y{i%n}({', '.join([f'y{y[i%n][j]}[{bs}]' for j,bs in enumerate(s)])})")

        for i,s in enumerate(cx):
            ribbon.append(f"CX{i}({', '.join([f'cx{s[j]}[.]' if j < len(s)-t else f'cx{s[j]}[{s[j]}]' for j in range(len(s))])})")

        for i,s in enumerate(cy):
            ribbon.append(f"CY{i}({', '.join([f'cy{s[j]}[.]' if j < len(s)-t else f'cy{s[j]}[{s[j]+(n*(n+t+t))}]' for j in range(len(s))])})")
    
    else:

        hxcore = np.array([row[t:-t] if t != 0 else row for row in hxfull]) # no extensions
        hycore = np.array([row[t:-t] if t != 0 else row for row in hyfull]) # no extensions

        hxcorel = hxcore.tolist()
        for i,s in enumerate(hxcorel[:int(n/2)-1]):
            hxcorel[i] = ['.' if j < int(n/2)-1-i else b for j,b in enumerate(s)]

        for i,s in enumerate(hxcorel[-int(n/2):]):
            hxcorel[hxcorel.index(s)] = [b if j < n-1-i else '.' for j,b in enumerate(s)]

        hxbs = [t*['.']+s for s in hxcorel]

        hycorel = hycore.tolist()
        for i,s in enumerate(hycorel[n+1:n+int(n/2)]):
            hycorel[hycorel.index(s)] = [b if j < n-1-i else '.' for j,b in enumerate(s)]

        for i,s in enumerate(hycorel[-int(n/2):]):
            hycorel[hycorel.index(s)] = ['.' if j < int(n/2)-i else b for j,b in enumerate(s)]

        hybs = [t*['.']+s for s in hycorel]

        ribbon = []

        for i,s in enumerate(hxbs):
            ribbon.append(f"X{i%n}({', '.join([f'x{x[i%n][j]}[{bs}]' for j,bs in enumerate(s)])})")

        for i,s in enumerate(hybs):
            ribbon.append(f"Y{i%n}({', '.join([f'y{y[i%n][j]}[{bs}]' for j,bs in enumerate(s)])})")

    params = []

    params.append(f"%var: 'inter_k_on' {inter_k_on}")
    params.append(f"%var: 'inter_k_off' {inter_k_off}")
    params.append(f"%var: 'intra_k_on' {intra_k_on}")
    params.append(f"%var: 'intra_k_off' {intra_k_off}")

    params.append(f"%var: 'inter_wobble_k_on' {inter_wobble_k_on}")
    params.append(f"%var: 'inter_wobble_k_off' {inter_wobble_k_off}")
    params.append(f"%var: 'intra_wobble_k_on' {intra_wobble_k_on}")
    params.append(f"%var: 'intra_wobble_k_off' {intra_wobble_k_off}")

    init = []
    init.append('%init: 0 ' + ', '.join([f'X{i}()' for i in range(n)] + [f'Y{i}()' for i in range(n)]))
    init.append(f'%init: {cslat_count} ' + ', '.join([f'CX{i}()' for i in range(len(cx))] + [f'CY{i}()' for i in range(len(cy))]))
    init.append(f"%init: 1 /*{len(ribbon)} agents*/ {', '.join(ribbon)}")

    mod = f'%mod: alarm {snap_interval} [true] do $SNAPSHOT "{fname}_sim.ka"; repeat [true]'

    defs = f'%def: "maxConsecutiveClash" "{maxConsecutiveClash}"'

    kappa_code = [defs] + [''] + agents + [''] + params + [''] + rules + [''] + init + [''] + [mod]

    with open(f'{directory}/{fname}_code.ka', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(kappa_code))

    print(f'Success! Generated {len(rules)} rules.')



# Functions below are for running simulations



def namescraper(directory,fname):
    '''
    Find the files with iteration numbers and a .ka extension in the directory 
    and return the part of the filename before the iteration number.
    '''
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.ka')]
    # Remove items in files if they contain the word "input"
    files = [f for f in files if 'input' not in f]
    files = [f for f in files if 'code' not in f]
    files = [f for f in files if f[-4] != 'm'] # removes unnumbered iterations
    stem = fname+'_sim_'
    iterations = []
    for i in range(len(files)):
        iterations.append(files[i][len(stem):-3])
    # Remove items in iterations if they can't be converted to an integer.
    iterations = [int(i) for i in iterations if i.isdigit()]
    iterations.sort()
    return stem, iterations


def maxsizer(directory,fname):
    stem, iterations = namescraper(directory,fname)
    maxsizes = []
    simtimes = []
    for snap_no in [0]+iterations:
        if snap_no == 0:
            snap = ks.SnapShot(f'{directory}/{stem[:-1]}.ka')
        else:
            snap = ks.SnapShot(f'{directory}/{stem}{snap_no}.ka')
        complexes = snap.complexes
        sizes = [complexes[i].size for i in range(len(complexes))]
        maxsizes.append(np.max(sizes))
        simtimes.append(snap.time)
        
    return maxsizes, simtimes


def runsim(simnum,directories,fname):

    directory = directories[simnum]
    maxtime = time.time() + 2*60*60 # set time limit for simulation to 2 hours
    # Create simulation package to only run initialization once as this is the same for all runs and can be quite slow with larger number of rules
    if simnum == 0:
        sim = subprocess.Popen(['KaSim', f'{fname}_code.ka', '-make-sim', f'{fname}.kasim', '-sharing', 'None'], cwd=directory)
    else:
        while time.time() < maxtime: # if running simulations in parallel, allow for some time to create simulation package once
            if os.path.exists(f'{directories[0]}/{fname}.kasim'):
                sim = subprocess.Popen(['KaSim', '-load-sim', f'../{fname}_0/{fname}.kasim', '-sharing', 'None'], cwd=directory)
                break
            else:
                time.sleep(2)
    print(f'{simnum} Simulation is running\n')
    
    while sim.poll() is None:
        if time.time() > maxtime:
            sim.kill()
            print(f'{simnum} Simulation timed out\n')
            return maxsizer(directory,fname)
        time.sleep(5) # check every x seconds
        try:
            maxsizes,simtimes = maxsizer(directory,fname)
            print(simnum, maxsizes[-1], '\n')
            for k in range(len(maxsizes)-1):
                if maxsizes[k+1] < (2/3)*maxsizes[k] or maxsizes[k+1] < maxsizes[0]:
                    # run for another x seconds and kill the simulation once the max complex size is smaller than the initial ribbon size
                    time.sleep(2)
                    sim.kill()
                    maxsizes,simtimes = maxsizer(directory,fname)
                    print(simnum, maxsizes, '\n')
                    return maxsizes,simtime
        except:
            print(f'{simnum} Failed to generate max size comparison\n')


def runparallelsims(simnums,cpus,directories,fname):

    for c in range(int(simnums/cpus)):
        jobs = []
        for i in range(cpus*c,cpus*(c+1)):
            p = Process(target=runsim, args=(i,directories,fname,))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()