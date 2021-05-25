from network import *

assert __name__ == '__main__', 'Main program startup error.'


def main():
    nn = Network()
    nn.create_test_structure(ReLU)
    print('Test dataset accuracy: ', nn.test(TEST))


main()
