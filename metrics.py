def loss_diff(train, test):
    return max(0, test - train)

def accuracy_diff(train, test):
    return max(0, train - test)