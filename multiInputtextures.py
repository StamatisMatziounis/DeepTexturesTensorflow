from deeptexturestf import DeepTexture
from PIL import Image

feature_layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool','var_loss']
name = 'tex'


losses = []
finalLosses = {}
instanceList = []
scoreList    = []


def createLoss():
    '''
            This is a helper function of this program that populates the finalLosses dictionary with the best scoring losses of each texture
    '''
    # if only one instance return 
    if(len(instanceList)<1):
        return None

    # Pick the best score out of the instance list
    for i in instanceList[0].layer_loss_scores.keys():
        min_ = 999999999999999999999999999999
        mintensor = None
        for j in range(len(instanceList)):
            if (instanceList[j].layer_loss_scores[i] < min_):
                min_ = instanceList[j].layer_loss_scores[i]
                mintensor = j

        if (mintensor == None):
            raise ValueError("Outstandingly large value for all losses, check texture image(s)")
        finalLosses[i] = mintensor
    print("final Loss indices:",finalLosses)

    # Removing images that are not used in the file
    unusedImages = []
    for j in range(len(instanceList)):
        if j not in set(finalLosses.values()):
            unusedImages.append(j)
            # Updating the indices to remove unused textures from final texture
            print(instanceList[j].tex_path,"not used for final texture calculation. Check the clarity of the image.")

    # Updating indices
    for i in finalLosses.keys():
        for j in unusedImages:
            if(finalLosses[i]>j):
                finalLosses[i] = finalLosses[i]-1
    
    return finalLosses,unusedImages

def calculateWeights():
    '''
            This is a helper function of this program that populates the finalLosses dictionary with the list of scores of each layer
    '''
    for i in feature_layers:
        old_scores = []
        
        # Populating the old scores list by obtaining all scores from the same layer
        for j in range(len(instanceList)):
            old_scores.append(instanceList[j].layer_loss_scores[i])
            print(instanceList[j].layer_loss_scores)
        
        # Calculating the Weighted Score and storing it to the lexicon. For more info check function comments
        newScores = calculateWeightedScore(old_scores)
        finalLosses[i] = newScores

    return finalLosses


def initializeList(base_path,tex_path_list,saveLoc_ = None):
    '''
            This is a simple helper function of this program that initializes and populates the instance list with DeepTexture instances
    '''
    for i in range(len(tex_path_list)):
        currentName = name+str(i+1)
        if(saveLoc_ == None):
            instanceList.append(DeepTexture( currentName, tex_path_list[i], base_img_path = base_path))
        else:
            instanceList.append(DeepTexture( currentName, tex_path_list[i], base_img_path = base_path,saveLoc_ = saveLoc_))
        scoreList.append(0)

def deleteInstanceList():
    '''
            This is a simple helper function of this program that frees up the memory taken by the instance list
    '''
    global instanceList
    del instanceList

def reset():
    global losses,finalLosses,instanceList,scoreList
    losses = []
    finalLosses = {}
    instanceList = []
    scoreList    = []

def buildTexturesWithLoss(features = 'pool'):
    '''
            This is a helper function of this program that builds the textures in series
    '''
    for i in instanceList:
        i.buildTextureFull(features,withLoss=1)

def buildTextures(features = 'pool'):
    '''
            This is a helper function of this program that builds the textures in series
    '''
    for i in instanceList:
        i.buildTextureFull(features)

def runIterations(iterations_ = 2,pInterval = 10,save = 20):
    '''
            This is a helper function of this program that runs DeepTexture iterations in series (regarding the instances)
    '''
    tempreturn = []
    for i in range(len(instanceList)):
        output_ = instanceList[i].runIterations(iterations = iterations_,printInterval = pInterval,save = save)
        scoreList[i] = output_[1]
        tempreturn.append(output_[0])
    return tempreturn

def calculateWeightedScore(scoreList,offset = 0.2):
    '''
            This is a helper function of this program that calculates the weighted score of each output photo, given the distinct scores
    '''
    # Initialization of new Score list
    newScore = []
    print(scoreList)
    # Getting min and max to convert from min <= oldscore <= max to offset <= newScore <= (1-offset) so the scores contribute at least 20% and at most 80%
    min_ = min(scoreList)
    max_ = max(scoreList)
    # Initialization of the variables for the conversion
    m = 1-2*offset
    l = offset/m
    # Converting to [offset,1-offset] using  (1-2offset) * ( (x-min)/(max-min) + (offset)/(1-2*offset) ) 
    # What is done is a conversion from [min_score,max_score] to [0,1] by (x-min_score)/(max_score-min_score)
    # Then an abstract_offset is added so that 0 becomes "offset" and 1 becomes "1-offset".
    # If the initial abstract offset is called y so it becomes [y,1+y], there has to be a variable k so that y/k = offset and (1+y)/k = 1-offset
    # The solution for the system [y/k,(1+y)/k] = [offset,1-offset] is: "y = (offset)/(1-2*offset)" and "k = 1/(1-2*offset)"
    # For convienience "m = 1/k", "y = offset/m" and "new_score =   m * (score_from_0_to_1 + y)"
    for i in scoreList:
        if(max_-min_== 0):
            newScore.append(1/len(scoreList))
        else:
            newScore.append(m*(((i-min_)/(max_-min_))+l))
    #print("newScore before sum conversion:",newScore)

    # Getting the sum of the scores
    sum_ = sum(newScore)
    # Weighing down the scores by the sum so sum(newScore) = 1
    # This procedure is done so that when the pictures merge, there is no clipping
    for i in range(len(newScore)):
        newScore[i]/=sum_
    return newScore

def printScores():
    '''
            Pretty self explanatory
    '''
    print("Average Scores are:",scoreList)


def calculateOutput(saveLoc = 'results/',imageoverride = None):
    '''
            This is a helper function that calculates the output image given the distinct output images and their score
    '''
    # Getting the weighted scores that any pixel will be weighed, then added
    newScore = calculateWeightedScore(scoreList)

    # Initialization of the created image
    images = []

    if (imageoverride == None):
        # Adding the images to the list
        for i in instanceList:
            images.append(Image.open(i.fname))
    else:
        for i in imageoverride:
            images.append(Image.open(i))

    # Creation of a modification matrix for each r g b value to multiply of each image by the weight
    for i in range(len(images)):
        matrix =   (newScore[i],    0,          0,            0,
                    0,              newScore[i],0,            0,
                    0,              0,          newScore[i],  0)
        images[i] = images[i].convert("RGB",matrix)
    
    # Getting the size and initializing the new image
    sizes = images[0].size
    new_image = Image.new('RGB',sizes,'black')

    # Getting the pixels "weird list of tuples" of the image
    pixels = new_image.load()

    # This assortment of ifs means "for each pixel on all images"
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            sum_ = [0,0,0]
            for s in range(len(images)):
                # Getting image's pixel and adding it to the sum pixel
                old_pixel = images[s].getpixel((i,j))
                sum_[0] += old_pixel[0]
                sum_[1] += old_pixel[1]
                sum_[2] += old_pixel[2]
            # The final pixel value is the sum of all the weighted pixel values from the images
            pixels[i,j] = (sum_[0],sum_[1],sum_[2])

    # Creating the filename and saving the image
    if(imageoverride == None):
        file_name = 'data/'+saveLoc+name+'_at_iteration %d.png'%(instanceList[0].total_iterations)
    else:
        file_name = 'data/tex_test.png'
    new_image.save(file_name)
    return

def getInput():
    '''
            This is a helper function that gets a positive input of an integer recursively for the program to run
    '''
    iterations = input()
    
    try:
        iterations = int(iterations)
        if (iterations < 0):
            raise ValueError()
    except:
        print("Invalid number, Reminder: The value must be a positive integer, try again: ",end = ' ')
        iterations = getInput()
    
    return iterations

def ruinsNormal(tex_img,base_img,features_ = "pool", iterations_ = None,printInterval_ = 50, save_ = 100):
    print("             Normal Texture Method                 ")
    finalNetwork = DeepTexture( (name+"_final"), tex_img, base_img_path = base_img,saveLoc_='resultsNorm/')
    finalNetwork.buildTextureFull(features = features_)
    
    print("Running Iterations...")
    print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
    if (iterations_== None):
        iterations_ = getInput()
        while (iterations_>0):
            time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)
            print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
            iterations_ = getInput()
    else:
        time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)
    print("Resetting...")
    reset()
    return time_

# Structure of the Minimum method of texture enhancement
def ruinsMin(tex_list,base_img, features_ = "pool",iterations_ = None,printInterval_ = 50,save_ = 100):
    
    print("             Minimum Texture Method                 ")
    # Initialization of the neural networks
    initializeList(base_img,tex_list)

    print("Gathering best scores...")
    # Building textures to determine the loss of each layer 
    buildTexturesWithLoss(features = features_)
    finalLosses,unusedImages = createLoss()

    # Deep Copying tex_list so other programs utilising the same tex_list won't be affected by popping
    tex_list2 = []
    for i in tex_list:
        tex_list2.append(i)
    
    # Removing unused images via sorting the indices from last to first so the indices of the remaining images don't change.
    unusedImages.sort(reverse=True)
    for i in unusedImages:
        tex_list2.pop(i)
    finalNetwork = DeepTexture( (name+"_final"), tex_list2, base_img_path = base_img,saveLoc_ = 'resultsMin/')
    
    deleteInstanceList()
    # Building final texture
    finalNetwork.buildTextureFull(features = features_,lossIndices = finalLosses,varLoss = 1)
    
    print("Scores Gathered, Running iterations")
    # Training the model
    if(iterations_ == None):
        print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
        iterations_ = getInput()
        while (iterations_>0):
            time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)[0]
            print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
            iterations_ = getInput()
    else:
            time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)[0]
    print("Resetting...")
    reset()
    return time_

# Example application of the Minimum method
def ruins1():
    '''
            This is a test program that runs a texture based out of 3 texture images and outputs the clean texture
    '''
    # Initialization of list of images to put through the program
    tex_list = ['data/inputs/tex_ruins1.png','data/inputs/tex_ruins3.png','data/inputs/tex_ruins4.png']
    base_img = 'data/inputs/base_ruins22.png'
    ruinsMin(tex_list,base_img)

# Example application of the Minimum method    
def ruins2():
    '''
            This is a test program that runs a texture based out of 3 texture images and outputs the clean texture
    '''
    # Initialization of list of images to put through the program
    tex_list = ['data/inputs/ruins4/tex_ruins1.png','data/inputs/ruins4/tex_ruins2.png','data/inputs/ruins4/tex_ruins6.png','data/inputs/ruins4/tex_ruins4.png']
    base_img = 'data/inputs/ruins4/base_ruins3.png'
    ruinsMin(tex_list,base_img)

# Example application of the Minimum method
def ruins3():
    '''
            This is a test program that runs a texture based out of 3 texture images and outputs the clean texture
    '''
    # Initialization of list of images to put through the program
    tex_list = ['data/inputs/ruins111/tex_ruins1.png','data/inputs/ruins111/tex_ruins22.png']
    base_img = 'data/inputs/ruins111/base_ruins.png'

    ruinsMin(tex_list,base_img)

# Structure of the Weighted Average method of texture enhancement
def ruinsWeightAVG(tex_list,base_img,features_ ="pool",iterations_ = None,printInterval_ = 50,save_ = 100):

    print("             Weighted Average Texture Method                 ")
    # Initialization of the neural networks
    initializeList(base_img,tex_list,saveLoc_='resultsWAVG/')
    
    print("Building all textures...")
    # Building textures to determine the loss of each texture
    buildTextures(features=features_)
    
    print("Textures built. Running Iterations...")
    if (iterations_ == None):
        # Running iterations to determine the score of each neural network
        print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
        iterations_ = getInput()
        while (iterations_>0):
            time_ = runIterations(iterations_,pInterval = printInterval_, save = save_)    
            print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
            iterations_ = getInput()    
    else:
        time_ = runIterations(iterations_,pInterval = printInterval_, save = save_)
    # Calculating output
    calculateOutput(saveLoc='resultsWAVG/')
    
    print("Resetting...")
    # Resetting the variables
    reset()
    deleteInstanceList()
    return time_

# Example application of the Weighted Average method
def ruinsWeightAVGrun():
    tex_list = ['data/inputs/tex_ruins1.png','data/inputs/tex_ruins3.png','data/inputs/tex_ruins4.png']
    base_img = 'data/inputs/base_ruins222.png'
    
    ruinsWeightAVG(tex_list,base_img)



# Structure of the Average method of texture enhancement
def ruinsAVG(tex_list,base_img,features_ = "pool",iterations_ = None,printInterval_ = 50,save_ = 100):
    print("             Average Texture Method                 ")
    finalNetwork = DeepTexture( (name+"_final"), tex_list, base_img_path = base_img,saveLoc_='resultsAVG/')
    finalNetwork.buildTextureFull(features = features_)
    
    print("Running Iterations...")
    print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
    if (iterations_== None):
        iterations_ = getInput()
        while (iterations_>0):
            time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)[0]
            print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
            iterations_ = getInput()
    else:
        time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)[0]
    print("Resetting...")
    reset()

# Example application of the Average method
def ruinsAVGrun():
    tex_list = ['data/inputs/tex_ruins1.png','data/inputs/tex_ruins3.png','data/inputs/tex_ruins4.png']
    base_img = 'data/inputs/base_ruins22.png'
    ruinsAVG(tex_list,base_img)



# Structure of the Weighted Average 2 method of texture enhancement
def ruinsWeightAVG2(tex_list,base_img,features_ = "pool",iterations_ = None,printInterval_ = 50,save_ = 100):
    
    print("             Weighted Average 2 Texture Method                 ")
    # Initialization of the neural networks
    initializeList(base_img,tex_list)
    
    print("Gathering best scores...")
    # Building textures to determine the loss of each layer 
    buildTexturesWithLoss(features = features_)
    
    print("Scores Gathered, Calculating Weights...")
    finalWeights = calculateWeights()

    deleteInstanceList()
    
    print("Building Final Texture...")
    finalNetwork = DeepTexture( (name+"_final"), tex_list, base_img_path = base_img,saveLoc_='resultsWAVG2/')
    finalNetwork.buildTextureFull(features = features_,lossIndices = finalWeights)
    
    print("Running iterations...")    
    if (iterations_== None):
        iterations_ = getInput()
        while (iterations_>0):
            time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)[0]
            print("Give the number of iterations that you want the program to run. (Type 0 to exit.):",end=" ")
            iterations_ = getInput()
    else:
        time_ = finalNetwork.runIterations(iterations = iterations_,printInterval = printInterval_,save=save_)[0]
    print("Resetting...")
    reset()
    return time_

# Example application of Weighted Average 2 method
def ruinsAVG3run():
    tex_list = ['data/inputs/ruins1/tex_ruins1.png','data/inputs/ruins1/tex_ruins3.png','data/inputs/ruins1/tex_ruins4.png']
    base_img = 'data/inputs/ruins1/base_ruins22.png'
    ruinsWeightAVG2(tex_list,base_img)


def evaluationOfMethods1():
    tex_list = ['data/inputs/ruins1/tex_ruins1.png','data/inputs/ruins1/tex_ruins3.png','data/inputs/ruins1/tex_ruins4.png']
    base_img = 'data/inputs/ruins1/base_ruins22.png'
    
    with open('evaluationTimes.txt','a') as f:
        
        temptime = ruinsNormal(tex_list[0],base_img,iterations_ = 20000)
        temp = "Normal: "+str(temptime)+"\n"
        f.write(temp)

        temptime = ruinsMin(tex_list,base_img,iterations_ = 20000)
        temp = "Minimum: "+str(temptime)+"\n"
        f.write(temp)

        #temptime = ruinsWeightAVG(tex_list,base_img,iterations_ = 20000)
        #temp = "Weighted Average: "+str(temptime)+"\n"
        #f.write(temp)

        temptime = ruinsAVG(tex_list,base_img,iterations_ = 20000)
        temp = "Average: "+str(temptime)+"\n"
        f.write(temp)

        temptime = ruinsWeightAVG2(tex_list,base_img,iterations_ = 20000)
        temp = "Weighted Average 2: "+str(temptime)+"\n"
        f.write(temp)
        f.close()
    return


if __name__ == '__main__':
    
    # Here is a bunch of manual tests 

    #tex = DeepTexture('tex1','data/inputs/tex_ruins2.png',base_img_path="data/inputs/base_ruins.png")
    #tex.buildTexture(features='all')
    #a = tex.runIterations(iterations = 2)
    #print("for tex we have loss:",a)
    #a = tex.runIterations(iterations = 4)
    #print("for texx we have loss:",a)
    
    # Runnning the iterations
    #runIterations(100,10)
    # Printing the scores
    #printScores()
    # Resuming by running more iterations
    #runIterations(100,10)
    # Printing final scores
    #printScores()
    #calculateOutput()

    evaluationOfMethods1()
