from sk import SKLearnClassification
from sk import SKLearnClassification


# print('Incepem cu SKLearn\n')

# sk_class = SKLearnClassification()
# sk_class.execute()
# sk_class.evaluate_multiple_splits()
#
# print('Incepem cu TensorFlow\n')
#
# tf_class = TensorFlowClassification()
# tf_class.execute()
#
# print('Incepem cu FaceNet\n')
#
# fn_class = FaceNetTorchClassification()
# fn_class.execute()
#
# print('Incepem cu VGG\n')
#
# vgg_class = VGGClassification()
# vgg_class.execute()

# print('Incepem cu Dlib\n')
#
# dlib_class = DlibClassification()
# dlib_class.execute()

def main():
    print('Evaluating Scikit-Learn PCA + SVM')

    classifier = SKLearnClassification()
    classifier.evaluate_multiple_splits()

    print('Other Algorithms')
    run_other_algorithms()


if __name__ == '__main__':
    main()
