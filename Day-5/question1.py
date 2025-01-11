chars= input("Enter a word: ")
result=0
for char in chars:
    if char.isalpha():
        num= ord(char.lower()) -97
        if num<4 :
            str='1'
        else:
            str='0'
        for i in range(num):
            str+= '0'
    else:
        str='0'
        print('Invalid character')
    result+= int(str)
print(result)