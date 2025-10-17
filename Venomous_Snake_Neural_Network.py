import csv
import numpy as np

# File Location of Program.
PATH = str(__file__).replace("Venomous_Snake_Neural_Network.py", "")

#  Weight Training Variables.
input_matrix = []
output_matrix = []
temp_input = []

record_on = False
index = 0
line_limit = 6


# Unload CSV Spreadsheet.
with open("Snakes_Dataset.csv", mode = "r") as file:
    
    file = csv.reader(file)
    
    for lines in file:
        
        if len(lines) > 0:
            
            for i in lines:

                # Start Recording Tags from File.
                if record_on == True:

                    # Increase Index.
                    index += 1

                    # Ignore Datatag Title.
                    if index > 1:

                        # Record Input Tag.
                        if index < line_limit:
                            temp_input.append(float(i[0]))

                        # Record Output Tag.
                        else:
                            input_matrix.append(temp_input)
                            output_matrix.append([float(i[0])])

                            # Reset Temp Input List.
                            temp_input = []

                    # Reset Index - New Tag.
                    if index == line_limit:
                        index = 0

                else:

                    # Finish with Variables Section.
                    if "#" in i:
                        record_on = True



# Neural Network Class.
class NeuralNetwork():
    def __init__(self):

        # Create Input / Output matrices.
        self.X = np.array(input_matrix)
        self.y = np.array(output_matrix)

        # Randomly initialize our weights with mean 0
        self.syn0 = 2 * np.random.random((len(self.X[0]), len(self.X))) - 1
        self.syn1 = 2 * np.random.random((len(self.y), len(self.y[0]))) - 1


    # Sigmoid + Derivative Function.
    def nonlin(self, x, deriv=False):
        if (deriv == True):
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    # Train Weights.
    def train(self, steps, data, change=True):
        for j in range(steps):

            # Feed forward through layers 0, 1, and 2
            l0 = data
            l1 = self.nonlin(np.dot(l0, self.syn0))
            l2 = self.nonlin(np.dot(l1, self.syn1))

            # How much did we miss the target value?
            l2_error = self.y - l2

            if (j % 1000) == 0:
                print("Error:" + str(np.mean(np.abs(l2_error))))
                print(str((j / steps) * 100) + "% complete.")

            # In what direction is the target value?
            l2_delta = l2_error * self.nonlin(l2, deriv=True)

            # How much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(self.syn1.T)

            # In what direction is the target l1?
            l1_delta = l1_error * self.nonlin(l1, deriv=True)

            if (change == True):
                self.syn1 += 0.05 * l1.T.dot(l2_delta)
                self.syn0 += 0.05 * l0.T.dot(l1_delta)

        # Get results.
        return (l2)


# Make a Prediction based on User Data.
def Test():

    # Get User Data Input for Prediction.
    test_matrix = input("Input Dataset for Prediction (e.x. 1,0,0,0): ")
    temp_matrix = []
    for x in test_matrix:
        if x != ",":
            temp_matrix.append(int(x))

    # Run the Test Matrix through the Neural Network.
    result = neural_network.train(1, temp_matrix, False)
    result = round(float(result.item()), 2) * 100

    print(" ")
    print(str(result) + "% Confidence of Venomosity")

    # Restart Function.
    Test()


print("TRAINING NEURAL NETWORK...")
neural_network = NeuralNetwork()
neural_network.train(100000, neural_network.X, True)
print("------")
print(" ")

# Start Testing.
Test()


    


