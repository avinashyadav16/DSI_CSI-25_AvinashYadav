"""
TASK: 
    Create lower triangular, upper triangular and pyramid containing the "*" character.


DESCRIPTION:
    This script prints three different triangle patterns using asterisks (*):
        1. Lower Triangular
        2. Upper Triangular
        3. Pyramid

    The size of the triangles is determined by user input. 
    Each pattern is displayed in the console output.


AUTHOR: Avinash Yadav

DATE: 08 June 2025
"""


def lower_triangle(numRows):
    for i in range(1, numRows + 1):
        print('* ' * i)


def upper_triangle(numRows):
    for i in range(numRows, 0, -1):
        print('* ' * i)


def pyramid(numRows):
    for i in range(1, numRows + 1):
        print(' ' * (numRows - i) + '* ' * i)


def main():
    try:
        numRows = int(input("Enter the number of rows: "))

        print("\nLower Triangular Pattern:")
        lower_triangle(numRows)

        print("\nUpper Triangular Pattern:")
        upper_triangle(numRows)

        print("\nPyramid Pattern:")
        pyramid(numRows)
    except ValueError:
        print("Please enter a valid integral value.")


if __name__ == "__main__":
    main()


'''

OUTPUT:

    Enter the number of rows: 5

    Lower Triangular Pattern:
    * 
    * * 
    * * * 
    * * * * 
    * * * * * 

    Upper Triangular Pattern:
    * * * * * 
    * * * * 
    * * * 
    * * 
    * 

    Pyramid Pattern:
        *
       * *
      * * *
     * * * *
    * * * * *
    
'''
