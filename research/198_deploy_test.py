from condense_and_test import CondenseAndTest
from cifar10_condensa_script import CIFAR10CondenseAndTest
import argparse

'''
Parse the command-line arguments
Can add 
'''
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--iterate',
                    nargs='+',
                    help='List of densities to iterate over and compress',
                    type=float,
                    required=False)

parser.add_argument('-s', '--step',
                    nargs='+',
                    help='Start, num iterations, and step values to compress for',
                    type=float,
                    required=False)

parser.add_argument('-nc', '--noCompress',
                    help='Dont run Condensa compression')

args = parser.parse_args()

iterate_list = [0.8, 0.5, 0.2]
# [start, end, step]
step_list = [0.1, 0.01, 0.001]
nc = False
if (args.noCompress):
    nc = True

# if (args.iterate):
#     for i in args.iterate:
#         tester = CIFAR10CondenseAndTest(i, nc)
#         # Change below for cutom testing
#         tester.runTheGuantlet()
# elif (args.step):
#     cnt = args.step[0]
#     decimals = str(args.step[2])
#     decimals = decimals[::-1].find('.')
#     print(decimals)
#     for i in range(int(args.step[1])):
#         tester = CIFAR10CondenseAndTest(cnt, nc)
#         # Change below for cutom testing
#         tester.runTheGuantlet()

#         cnt -= args.step[2]
#         cnt = round(cnt, int(decimals))
# else:
#     raise Exception('Need to provide type of test\n'+
#                     'For help run the program with -h or --help')

base_model_shap = CIFAR10CondenseAndTest(0.8, True).explainBase()


# '''
# Declare program variables and objects
# '''
# density = args.density

# test_cnt = 0.085
# for i in range(100):
#     tester = CondenseAndTest(test_cnt)

#     '''
#     Run Tests
#     1. Condense
#     2. Run SHAP
#     3. Count Non Zero Paramters
#     4. Test Accuracy
#     '''
#     tester.condenseNeuron()
#     tester.runShap()
#     tester.countZeroWeights()
#     tester.testAccuracy()
#     tester.end()

#     test_cnt -= 0.001
#     test_cnt = round(test_cnt, 3)

# test_cnt = 0.09
# for i in range(90):
#     tester = CondenseAndTest(test_cnt)

#     '''
#     Run Tests
#     1. Condense
#     2. Run SHAP
#     3. Count Non Zero Paramters
#     4. Test Accuracy
#     '''
#     tester.condenseNeuron()
#     tester.runShapWithPercentages()
#     tester.write_to_shap()

#     test_cnt -= 0.001
#     test_cnt = round(test_cnt, 3)

""" put into gif folder """

# test_cnt = 0.1
# for i in range(99):
#     tester = CondenseAndTest(test_cnt)

#     '''
#     Run Tests
#     1. Condense
#     2. Run SHAP
#     '''
#     tester.condenseNeuron()
#     tester.runShapWithPercentages()

#     test_cnt -= 0.001
#     test_cnt = round(test_cnt, 3)




