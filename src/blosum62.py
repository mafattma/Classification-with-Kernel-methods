# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:14:08 2019

@author: fatma
"""
def blosum62_matrix():
    
    with open('blosum62.txt') as matrix_file:
      matrix = matrix_file.read()
    lines = matrix.strip().split('\n')

    header = lines.pop(0)
    columns = header.split()
    matrix = {}

    for row in lines:
      entries = row.split()
      row_name = entries.pop(0)
      matrix[row_name] = {}
      for column_name in columns:
        matrix[row_name][column_name] = entries.pop(0)
    
    return(matrix)
