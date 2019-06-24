import numpy as np
from scipy.special import softmax
import random
# extract data from relevat files
def extractFromFile():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    return train_x, train_y
# return normalized data for learning purposes
def getNormedTrainSet():
    objects, labels = extractFromFile()
    normed_objects = []
    numFeatures = len(objects[0])
    for object in objects:
        normalized = []
        for i in range(numFeatures):
            normalized.append((object[i] / 255))
        normed_objects.append(np.array(normalized))
    pairs = list(zip(normed_objects, labels))
    random.seed(3)
    random.shuffle(pairs)
    return pairs
# return normalized test set
def getNormedTestSet():
    toNorm = np.loadtxt("test_x")
    toRet = []
    numFeatures = len(toNorm[0])
    for object in toNorm:
        normalized = []
        for i in range(numFeatures): # calculate z score for each feature in current object
            normalized.append((object[i] / 255))
        toRet.append(np.array(normalized))
    return toRet
# learning set size is 55,000, thus we'll slice it into 5 pieces and return a list of pairs of (train_set, validate_set), such that train_set is 80% and validate_set is 20%
def slice(dataSet):
    toRet = []
    l1 = dataSet[:11000] # 1st part
    l2 = dataSet[11000:22000] # 2nd part
    l3 = dataSet[22000:33000] # 3rd part
    l4 = dataSet[33000:44000] # 4th part
    l5 = dataSet[44000:] # 5th part
    pair1 = (l1, l2+l3+l4+l5) # validate on 1st, learn on rest
    pair2 = (l2, l1+l3+l4+l5) # validate on 2nd, learn on rest
    pair3 = (l3, l1+l2+l4+l5) # validate on 3rd, learn on rest
    pair4 = (l4, l1+l2+l3+l5) # validate on 4th, learn on rest
    pair5 = (l5, l1+l2+l3+l4) # validate on 5th, learn on rest
    toRet.append(pair1)
    toRet.append(pair2)
    toRet.append(pair3)
    toRet.append(pair4)
    toRet.append(pair5)
    return toRet

#the ReLU function.
ReLU = lambda x: np.maximum(0, x)

class Neural_Network():

    def __init__(self, hidden_size, eta, epochs):
        self.__eta = eta
        self.__epochs = epochs
        self.__layers = self.init_net([784, hidden_size, 10]) # layer dimensions

    def init_net(self, dimsOfEachLayer):
       # list of wi and biases.
        neo_net = []
        for layerNum in range(1, len(dimsOfEachLayer)): #no layer 0.
        # there are diffrent ways of init epsilon.
            W = np.random.uniform(-0.08, 0.08, [dimsOfEachLayer[layerNum], dimsOfEachLayer[layerNum - 1]]) * 0.01
            b = np.random.rand(dimsOfEachLayer[layerNum])         
            # Add new params to the list
            neo_net.append([W, b])
        return neo_net

    
    def train(self,train_data):
        for epoch in range(self.__epochs):
            # total loss for epoch
            epoch_loss = 0
            for element in train_data:
                y = int(element[1])
                # Compute loss and the gradients
                loss, grads = self.bprop(element[0], y)
                epoch_loss += loss
                # update the parameters according to the gradients
                # and the learning rate.
                grads[0][1] = grads[0][1].reshape((grads[0][1].shape[0] , 1))
                grads[1][1] = grads[1][1].reshape((grads[1][1].shape[0] , 1))
                self.__layers[0][1] = self.__layers[0][1].reshape(self.__layers[0][1].shape[0] , 1)
                self.__layers[1][1] = self.__layers[1][1].reshape(self.__layers[1][1].shape[0] , 1)
                for i in range(len(self.__layers)):
                    self.__layers[i][0] = self.__layers[i][0] * ( 1 - 0.00001 * self.__eta) - self.__eta * grads[i][0]
                    self.__layers[i][1] -= self.__eta * grads[i][1]

    def midNorm(self, toNorm):
    # normalize (max division) temporary vector created as input for hidden layer
        max = np.max(toNorm)
        if max == 0:    # don't divide by zero!
            return toNorm
        return np.array([x / max for x in toNorm])

    def forward(self, input):
    #neo_net = w1,bias1,w2,bias2.... wn*biasn.
    # Compute all layers from input to output and put results in a list.
    # returns a list: [z1,z2...zn] for each layer.
        z_list = [input] #we make a list of vectors. the first entry is our inputed vector.
        counter = 0
        for layer in self.__layers: #because every wi and bias represt 1 layer.
        # Get layer.
            W, b = layer[0], layer[1]
            temp = np.dot(W, z_list[-1])
            b = np.squeeze(b)
            next_z = np.add(temp , b)
            if counter != 1:
                next_z = ReLU(next_z)
                next_z = self.midNorm(next_z)
            # Add current layer output.
            z_list.append(next_z)
            counter += 1
        # Softmax last output.
        z_list[-1] = softmax(z_list[-1])
        return z_list

    #predict sample's class.
    def predict(self, sample):
         probs = self.forward(sample)[-1] #put the sample throught the network.
         # Find argmax
         prediction = np.argmax(probs)
         return prediction

    def bprop(self,sample,label):
        grads = []
        z_array = self.forward(sample) #vector of [x,z1,z2]
        output_vector = z_array[-1] #last vector in the net. the output after softmax.
        loss = -np.log(output_vector[label]  + 0.000000001)
        dl_dz2 = output_vector.copy() #calculate the loss / x.
        dl_dz2[label] -= 1 
        z_array.pop() #take the output out of the stack. z2 out
        layers_copy = self.__layers.copy()
        layers_copy.reverse() #reverse for comfort.[w2,b2]first[w1,b1]second
        #calculate the bias and weights gradiants.
        dl_db2 = dl_dz2
        z1 = z_array.pop()
        h1 = ReLU(z1)
        h1Transpose = h1.reshape((1,h1.shape[0])) 
        dl_dz2 = dl_dz2.reshape((dl_dz2.shape[0], 1))
        dl_dw2 = np.dot(dl_dz2, h1Transpose)
        grads.append([dl_dw2, dl_db2])
        x = z_array.pop()
        w2,b2 = layers_copy[0]
        w1,b1 = layers_copy[1]
        zeros = np.zeros((z1.shape[0], 1))
        z1 = z1.reshape((z1.shape[0], 1))
        max = np.maximum(zeros, np.sign(z1))
        dl_dz1 = max * np.dot(w2.T, dl_dz2)
        dl_db1 = dl_dz1
        xTranspose = x.reshape((1, x.shape[0]))
        dl_dw1 = np.dot(dl_dz1, xTranspose)
        grads.append([dl_dw1, dl_db1])
        layers_copy.pop(0) #remove the first element in the list.
        grads.reverse() #reverse the grads list back. 
        return loss, grads
    #returns loss, [[dw1,db1],[dw2,db2]]

# cross validation with 20% validation and 80% learning sets
def crossValidation(train_set, hidden_size, eta, epochs):
    print("eta = %f, epochs = %d, hidden layer size = %d" % (eta, epochs, hidden_size))
    pairs_for_cross = slice(train_set)
    results = []
    for pair in pairs_for_cross:
        nn = Neural_Network(hidden_size, eta, epochs)
        train_data = pair[1]
        validation_data = pair[0]
        print("learning...")
        nn.train(train_data)
        bad_predicts = 0
        total = len(validation_data)
        print("validating...")
        for validation_pair in validation_data:
            x = validation_pair[0]
            y = int(validation_pair[1])
            if nn.predict(x) != y:
                bad_predicts += 1
        success_rate = (total - bad_predicts) / total
        print ("currect success rate = %f" % success_rate)
        results.append(success_rate)
    avg = sum(results) / len(results)
    print("average success rate = %f" % avg)
# give prediction on each object in the test set all predictions into a file line by line
def predictOnTest(nn, test_set):
    predictions = []
    for x in test_set:
        predictions.append(nn.predict(x))
    file = open("test_y", "w")
    first_predict = True
    for prediction in predictions:
        if first_predict == True:
            file.write(str(prediction))
            first_predict = False
        else:
            file.write("\n")
            file.write(str(prediction))
    file.close()

if __name__ == '__main__':
    pairs = getNormedTrainSet()
    nn = Neural_Network(100, 0.01, 50)
    nn.train(pairs)
    test_set = getNormedTestSet()
    predictOnTest(nn, test_set)