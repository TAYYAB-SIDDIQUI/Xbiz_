def my_decorator(func):
    def wrapper():
        print("Something before the function runs")
        func()
        print("Something after the function runs")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

def mydeco(fn):
    def wrap():
        print("Something 1")
        fn()
        print("something 2")
    return wrap

@mydeco
def new():
    print("something 3")

new()
