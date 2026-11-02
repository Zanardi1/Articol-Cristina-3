from multi_split import main as run_other_algorithms
from sk import SKLearnClassification


def main():
    print('Evaluating Scikit-Learn PCA + SVM')

    classifier = SKLearnClassification()
    classifier.evaluate_multiple_splits()

    print('Other Algorithms')
    run_other_algorithms()


if __name__ == '__main__':
    main()
