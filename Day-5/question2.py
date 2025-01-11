import math

def decompose_into_squares(n):
    result = []
    while n > 0:
        largest_square = int(math.sqrt(n)) ** 2
        result.append(largest_square)
        n -= largest_square
    return result
def decompose_until_ones(squares):
    steps = []
    while any(s > 1 for s in squares):
        new_squares = []
        for s in squares:
            if s > 1:
                new_squares.extend(decompose_into_squares(s))
            else:
                new_squares.append(s)
        steps.append(new_squares)
        squares = new_squares
    return steps



a = input("Enter a Number:")

if a.isnumeric():
    num = int(a)
    squares = decompose_into_squares(num)
    formatted_squares = '+'.join([f"{int(math.sqrt(s))}^2" for s in squares])
    print(f"The decomposition of {num} is {formatted_squares}")
    steps = decompose_until_ones(squares)
    for i, step in enumerate(steps, 1):
        formatted_step = '+'.join([f"{int(math.sqrt(s))}^2" for s in step])
        print(f"Step {i}: {formatted_step}")
else:
    print("Please enter a valid number")

