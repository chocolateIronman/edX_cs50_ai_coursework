import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    with open(filename) as f:
        # read the file into a dictionary
        reader = csv.DictReader(f)
        # for each row
        for row in reader:
            # split into evidence and labels
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                int(get_month(row['Month'])),
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                int(get_visitor(row['VisitorType'])),
                int(get_weekend(row['Weekend']))
            ])
            labels.append(get_label(row['Revenue']))
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # k-nearest neighbour model with nearest neighbours to look at = 1
    model = KNeighborsClassifier(n_neighbors=1)
    # fit the data
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive = 0
    negative = 0
    truePositive = 0
    trueNegative = 0
    for label, prediction in zip(labels, predictions):
        # positive label detected
        if label == 1:
            positive += 1
            # label matches prediction
            if label == prediction:
                truePositive += 1
        # negative label detected
        if label == 0:
            negative += 1
            # label matches prediction
            if label == prediction:
                trueNegative += 1
    # claclulating sensitivity and specificity
    sensitivity = truePositive / positive
    specificity = trueNegative / negative
    return (sensitivity, specificity)


def get_month(month):
    match month:
        case 'Jan':
            return 0
        case 'Feb':
            return 1
        case 'Mar':
            return 2
        case 'Apr':
            return 3
        case 'May':
            return 4
        case 'June':
            return 5
        case 'Jul':
            return 6
        case 'Aug':
            return 7
        case 'Sep':
            return 8
        case 'Oct':
            return 9
        case 'Nov':
            return 10
        case 'Dec':
            return 11


def get_visitor(visitor):
    match visitor:
        case 'Returning_Visitor':
            return 1
        case 'New_Visitor':
            return 0
        case _:
            return 0


def get_weekend(weekend):
    match weekend:
        case 'TRUE':
            return 1
        case 'FALSE':
            return 0


def get_label(revenue):
    match revenue:
        case 'TRUE':
            return 1
        case 'FALSE':
            return 0


if __name__ == "__main__":
    main()
