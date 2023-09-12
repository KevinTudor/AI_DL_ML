# -*- coding: utf-8 -*-
"""
Introduction to Deep Learning – CAP 4613
Assignment 1
Due: Sunday, Jan 23 before 11:59 pm

Name: Kevin Tudor z-number:Z23468207

Spyder (Python 3.8) IDE

link to final version in Colab:
https://colab.research.google.com/drive/1FzvNVOBMhSEHh0_w2nKbnj8ae3tMAoTJ?usp=sharing
"""
import webbrowser
import math

#url = 'https://colab.research.google.com/drive/1FzvNVOBMhSEHh0_w2nKbnj8ae3tMAoTJ?usp=sharing'
#webbrowser.open(url)  # Go to the google colab

"""
Problem 1) 
Simple Calculator: 
In Python, implement a simple calculator that does the following 
operations: summation, subtraction, multiplication, division, mod, power, exp, 
natural log, and abs. 
"""

#PART 1.A)
 
def calculator():
    
    def __sum(x, y):
        return x + y
    def __sub(x, y):
        return x - y
    def __mul(x, y):
        return x * y
    def __div(x, y):
        return x / y
    def __mod(x, y):
        return x % y
    def __pow(x, y):
        return x ** y
    def __exp(x):
        return math.exp(x)
    def __log(x):
        return math.log(x)
    def __abs(x):
        if type(x) == float:
            return math.sqrt(x ** 2)
        else:
            return int(math.sqrt(x ** 2))
    
    print("Simple calculator of following operations:\
          \n\nsummation (+)\
          \nsubtraction (-)\
          \nmultiplication (*)\
          \ndivision (/)\
          \nmodulus (%)\
          \npower (**)\
          \nexponential (^)\
          \nnatural log (ln)\
          \nabsolute value (||)") 
          
    f_num = 0
    s_num = 0
          
    while f_num != 'x':      
        #ask the user to enter: first number, operation, and second number (if required).
      
        print("\nEnter the first number:")
        f_num = input()
        
        if f_num.lstrip("-").isdigit():
            f_num = int(f_num)
            print(type(f_num), f_num)
        else:
          try:
            f_num = float(f_num)
            print(type(f_num), f_num)
          except ValueError:
            if f_num == 'x':
                print("Program Quiting")
                break
            print(f_num, "is not digit or float, defaulting first number to 0")    
            f_num = 0 #default to 0
        
        print("\nEnter the operation:")
        oper = input()
        if oper.isalpha():
            oper = oper.lower()
            if oper == 'x':
                break
      
        if oper == "^":
            print("\nNo second number needed")
            print(f_num, oper, "=", __exp(f_num))
            
        elif oper == "ln":
            print("\nNo second number needed")
            print(f_num, oper, "=", __log(f_num))
            
        elif oper == "||":
            print("\nNo second number needed")
            print("|", f_num, "|", "=", __abs(f_num))
        else:
            print("\nEnter the second number:")
            s_num = input()
            
            if s_num.lstrip("-").isdigit():
                s_num = int(s_num)
            else:
                try:
                    s_num = float(s_num)
                    print(type(s_num), s_num)
                except ValueError:
                    if s_num == 'x':
                        print("Program Quiting")
                        break
                    print(s_num, "is not digit or float, defaulting second number to 0")    
                    s_num = 0 #default to 0
            
            if oper == "+":
                print(f_num, oper, s_num, "=",__sum(f_num, s_num))
            elif oper == "-":
                print(f_num, oper, s_num, "=",__sub(f_num, s_num))
            elif oper == "*":
                print(f_num, oper, s_num, "=",__mul(f_num, s_num))
            elif oper == "/":
                print(f_num, oper, s_num, "=",__div(f_num, s_num))
            elif oper == "%":
                print(f_num, oper, s_num, "=",__mod(f_num, s_num))
            elif oper == "**":
                print(f_num, oper, s_num, "=",__pow(f_num, s_num))
            else:
                print("Unknown operator: ", oper)


""" 
PART 1.B)
        
******TEST RESULTS******

Simple calculator of following operations:          

summation (+)          
subtraction (-)          
multiplication (*)          
division (/)          
modulus (%)          
power (**)          
exponential (^)          
natural log (ln)          
absolute value (||)

Enter the first number:
-5
<class 'int'> -5

Enter the operation:
+

Enter the second number:
5
-5 + 5 = 0

Enter the first number:
-5
<class 'int'> -5

Enter the operation:
-

Enter the second number:
-5
-5 - -5 = 0

Enter the first number:
2.1
<class 'float'> 2.1

Enter the operation:
*

Enter the second number:
2
2.1 * 2 = 4.2

Enter the first number:
2.2
<class 'float'> 2.2

Enter the operation:
/

Enter the second number:
2
2.2 / 2 = 1.1

Enter the first number:
4
<class 'int'> 4

Enter the operation:
%

Enter the second number:
2
4 % 2 = 0

Enter the first number:
2
<class 'int'> 2

Enter the operation:
**

Enter the second number:
3
2 ** 3 = 8

Enter the first number:
3
<class 'int'> 3

Enter the operation:
^

No second number needed
3 ^ = 20.085536923187668

Enter the first number:
1
<class 'int'> 1

Enter the operation:
Ln

No second number needed
1 ln = 0.0

Enter the first number:
-3.3
<class 'float'> -3.3

Enter the operation:
||

No second number needed
| -3.3 | = 3.3

Enter the first number:
a
a is not digit or float, defaulting first number to 0

Enter the operation:
+

Enter the second number:
2
0 + 2 = 2

Enter the first number:
x
Program Quiting

"""

"""
Problem 2) 
Threshold-based Classifier: 
We have a two-class classification problem (i.e., C1 and C2). 
Each data sample is represented by two attributes (x, y). 
The three data samples in class C1 are:{(1, 1), (3, 2), (2, 3)}
and {(1, 2), (2, 2), (2, 1)} in class C2.
"""

def classifier():
    
    #import numpy as np
    import matplotlib.pyplot as plt
    
    C1 = {(1, 1), (3, 2), (2, 3)}
    C2 = {(1, 2), (2, 2), (2, 1)}
    
    
    """
    PART 2.A)
    Plot the data samples. 
    The data points in classes C1 and C2 must be in two different colors 
    and shapes. Label the axes and add legends as appropriate.
    """
    
    def take_fir(elem):
        return elem[0]

    def plot(C1, C2):
        
        #sort by x element in ascending order 
        C1_sorted = sorted(C1, key = take_fir)
        C2_sorted = sorted(C2, key = take_fir)
        
        #lists of x and y values
        x_lst1 = []
        y_lst1 = []
        
        x_lst2 = []
        y_lst2 = []
        
        for element in C1_sorted:
            x_lst1.append(element[0])
            y_lst1.append(element[1])
            
        #print("x_lst1: ", x_lst1, "\ny_lst1: ", y_lst1)
        
        for element in C2_sorted:
            x_lst2.append(element[0])
            y_lst2.append(element[1])
            
        #print("x_lst2: ", x_lst2, "\ny_lst2: ", y_lst2)
            
        plt.scatter(x_lst1, y_lst1, c = '#3399FF')
        plt.scatter(x_lst2, y_lst2, c = '#36FF33')
        plt.title('Threshold-based Classifier')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.legend(['C1', 'C2']) 
        plt.show()    
    
    
    plot(C1, C2)
        








#calculator()
classifier()















