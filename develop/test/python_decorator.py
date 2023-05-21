"""
파이썬 데코레이션 실습
"""

import tensorflow as tf

def decorator(func):
    def decorated():
        data = func()
        
        print("hello")
        print(data)
        print("bye")
        return data
    return decorated# 반환의 형태가 함수 핸들이어야 한다. 대코래이션 된 결과가 함수로써 실행가능해야 하는 듯.
    #return data# 에러 발생


@decorator
def function_test():
    return 3

print(function_test())

