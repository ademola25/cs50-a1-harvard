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
    
    # Month name to index mapping (handle both abbreviated and full names)
    month_to_index = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11,
        'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
        'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11
    }
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        
        for row in reader:
            # Parse each column according to the specification
            evidence_row = []
            
            # Administrative (int)
            evidence_row.append(int(row[0]))
            
            # Administrative_Duration (float)
            evidence_row.append(float(row[1]))
            
            # Informational (int)
            evidence_row.append(int(row[2]))
            
            # Informational_Duration (float)
            evidence_row.append(float(row[3]))
            
            # ProductRelated (int)
            evidence_row.append(int(row[4]))
            
            # ProductRelated_Duration (float)
            evidence_row.append(float(row[5]))
            
            # BounceRates (float)
            evidence_row.append(float(row[6]))
            
            # ExitRates (float)
            evidence_row.append(float(row[7]))
            
            # PageValues (float)
            evidence_row.append(float(row[8]))
            
            # SpecialDay (float)
            evidence_row.append(float(row[9]))
            
            # Month (convert to index 0-11)
            evidence_row.append(month_to_index[row[10]])
            
            # OperatingSystems (int)
            evidence_row.append(int(row[11]))
            
            # Browser (int)
            evidence_row.append(int(row[12]))
            
            # Region (int)
            evidence_row.append(int(row[13]))
            
            # TrafficType (int)
            evidence_row.append(int(row[14]))
            
            # VisitorType (1 for Returning_Visitor, 0 otherwise)
            evidence_row.append(1 if row[15] == 'Returning_Visitor' else 0)
            
            # Weekend (1 for TRUE, 0 for FALSE)
            evidence_row.append(1 if row[16] == 'TRUE' else 0)
            
            evidence.append(evidence_row)
            
            # Revenue label (1 for TRUE, 0 for FALSE)
            labels.append(1 if row[17] == 'TRUE' else 0)
    
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create a k-nearest neighbor classifier with k=1
    model = KNeighborsClassifier(n_neighbors=1)
    
    # Fit the model on the training data
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
    # Count true positives, false positives, true negatives, false negatives
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1
    
    # Calculate sensitivity (true positive rate)
    # Sensitivity = TP / (TP + FN)
    sensitivity = true_positives / \
        (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate specificity (true negative rate)
    # Specificity = TN / (TN + FP)
    specificity = true_negatives / \
        (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()