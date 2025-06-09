#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 11:20:36 2025

@author: mseifer1
"""

import numpy as np

def solve_wc(g, B, P):
    # g = gamma
    # B = beta
    # P = beta dot nhat
    # Solve quadratic: -wc^2 + 1 + g(Bwc + P)^2 = 0
    a = -1 + g * B**2
    b = -2 * g * B * P
    c = g * P**2 + 1

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None  

    sqrt_disc = np.sqrt(discriminant)
    wc1 = (-b + sqrt_disc) / (2 * a)
    wc2 = (-b - sqrt_disc) / (2 * a)
    return wc1, wc2
   
    #todo: if discriminant becomes negtaive set statement to check
    #sometimes theres both ++ or -- when large deviations
    #5 total, 2 scalars and a vector-> refactor so that accepts parameters in expected form

def celes_coords(vector):
    x, y, z = vector
    r = np.linalg.norm(vector) #r^2 = sqrt(x^2 + y^2 + z^2)
    dec_rad = np.arcsin(z / r)
    ra_rad = np.arctan2(y, x)

    # Convert to degrees and hours
    dec_deg = np.degrees(dec_rad)  #keep in radians 
    
    ra_deg = np.degrees(ra_rad)
    if ra_deg < 0:
        ra_deg += 360
    ra_hours = ra_deg / 15

    return ra_hours, dec_rad

#example values
g = 0.5
B = 1
n_hat = np.array([1, 1, 1])
n_hat = n_hat / np.linalg.norm(n_hat)  #normalize to unit vector
P = np.dot(B * n_hat, n_hat)  #B dot nhat

        
# solve
roots = solve_wc(g, B, P)
if roots:
    for i, wc in enumerate(roots, 1):
        if wc > 0:  #only positive wc 
            wc_vector = wc * n_hat
            ra, dec = celes_coords(wc_vector)
            print(f"Root {i}: wc = {wc:.4f}, RA = {ra:.2f}h, Dec = {dec:.2f}radians")




