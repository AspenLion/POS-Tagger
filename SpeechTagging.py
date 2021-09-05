# Project is a part-of-speech (POS) tagger using hidden Markov models.
# Utilizes naive Bayes for probability.
# Project was created using a private corpus. The corpus followed the following
# format:
#   1. txt file.
#   2. Each word is tagged.
#   3. Tagging followed the form: word=tag.
#   4. The tag set used is based on the following: https://github.com/slavpetrov/universal-pos-tags
#
# Note that the tags specify the POS for each word. Puctuation is
# also tagged as its own part of speech.
#
# Program Utilization:
#   1. Use load_corpus to load the dataset.
#   2. Create a Tagger object using the loaded dataset as the argument.
#   3. Create a sentence as a list where each word and punctuation is an element.
#   4. Call either Tagger methods with the word list as the argument.

# Imports

import math

# Load the data from the corpus.
def load_corpus(path):
    f = open(path, "r")
    lines = f.readlines()
    solution = []
    for i in lines:
        # Separate pairings
        POS_tagged = i.split()
        tupleForm = []
        for j in POS_tagged:
            # Separate tokens from tags (token,tag)
            removeEq = tuple(j.split("="))
            tupleForm.append(removeEq)
        solution.append(tupleForm)
    f.close()
    return solution

class Tagger(object):
    def __init__(self, sentences):
        # Dicts for tagToTag and tagToToken are implemented as
        #   nested dicts, where if 1->2, it is stored as dict[1][2]
        smoothing = 1e-5
        tagCountDict = {}
        initTagDict = {}
        initCount = len(sentences)
        tagToTagDict = {}
        tagToTokenDict = {}
        # Token -> Tag memory. Remembers all the tags that can
        #   belong to the token. Dict of list.
        tokenTotalTags = {}
        # Set up data into dictionaries for future calculations
        for i in sentences:
            # Initial tag
            if i[0][1] in initTagDict:
                initTagDict[i[0][1]] += 1
            else:
                initTagDict[i[0][1]] = 1
            # prevTag variable keeps track of the previous tag for
            #   tag->tag. Initialize to "" to avoid errors with
            #   the inital word (first word doesn't look for a prev).
            #   Also, note that the last word doesn't have an entry.
            prevTag = ""
            # Iterate through each word in the sentence
            for j in i:
                # Tag to tag
                if prevTag != "":
                    if prevTag in tagToTagDict:
                        if j[1] in tagToTagDict[prevTag]:
                            # Increment entry
                            tagToTagDict[prevTag][j[1]] += 1
                        else:
                            # First tag exists, but second doesn't
                            tagToTagDict[prevTag][j[1]] = 1
                    else:
                        # First tag doesn't exist
                        tagToTagDict[prevTag] = {j[1]:1}
                # Set new prevTag
                prevTag = j[1]
                # Tag to token
                if j[1] in tagToTokenDict:
                    if j[0] in tagToTokenDict[j[1]]:
                        # Increment entry
                        tagToTokenDict[j[1]][j[0]] += 1
                    else:
                        # Tag exists, but not token
                        tagToTokenDict[j[1]][j[0]] = 1
                else:
                    # Tag doesn't exist
                    tagToTokenDict[j[1]] = {j[0]:1}
                # Add to tokenTotalTags if not there
                if j[0] in tokenTotalTags:
                    temp = tokenTotalTags[j[0]]
                    temp.add(j[1])
                    tokenTotalTags[j[0]] = temp
                else:
                    tokenTotalTags[j[0]] = {j[1]}
        # Vocab size for the nested dicts (needed for probability)
        vocabTagToTag = 0
        for i in tagToTagDict:
            vocabTagToTag += len(i)
        vocabTagToToken = 0
        for i in tagToTokenDict:
            vocabTagToToken += len(i)
        # Set as self variables in case needed
        self.initTagDict = initTagDict
        self.tagToTagDict = tagToTagDict
        self.tagToTokenDict = tagToTokenDict
        self.tokenTotalTags = tokenTotalTags
        # Now that all dicts are found, use formula to find log probs
        initTagProb = {}
        tagToTagProb = {}
        tagToTokenProb = {}
        # Init
        for i in initTagDict:
            initTagProb[i] = math.log((initTagDict[i]+smoothing)/(initCount+(smoothing*(len(initTagDict)+1))))
        initTagProb["<UNK>"] = math.log(smoothing/(initCount+(smoothing*(len(initTagDict)+1))))
        # Turn UNK into a tag (tag->UNK)
        #   Note that UNK->tag doesn't make sense
        # TagToTag
        for i in tagToTagDict:
            # Limit total count to be conditional on previous tag
            count = 0
            vocab = len(tagToTagDict[i])
            for j in tagToTagDict[i]:
                count += tagToTagDict[i][j]
            for j in tagToTagDict[i]:
                if i in tagToTagProb:
                    # Previous tag exists
                    tagToTagProb[i][j] = math.log((tagToTagDict[i][j]+smoothing)/(count+(smoothing*(vocab+1))))
                else:
                    # Previous tag doesn't exist
                    tagToTagProb[i] = {j:math.log((tagToTagDict[i][j]+smoothing)/(count+(smoothing*(vocab+1))))}
            tagToTagProb[i]["<UNK>"] = math.log(smoothing/(count+(smoothing*(vocab+1))))
        # tag->UNK, for UNK = token
        # TagToToken
        for i in tagToTokenDict:
            # Limit total count to be conditional on tag
            count = 0
            vocab = len(tagToTokenDict[i])
            for j in tagToTokenDict[i]:
                count += tagToTokenDict[i][j]
            for j in tagToTokenDict[i]:
                if i in tagToTokenProb:
                    # Previous tag exists
                    tagToTokenProb[i][j] = math.log((tagToTokenDict[i][j]+smoothing)/(count+(smoothing*(vocab+1))))
                else:
                    # Previous tag doesn't exist
                    tagToTokenProb[i] = {j:math.log((tagToTokenDict[i][j]+smoothing)/(count+(smoothing*(vocab+1))))}
            tagToTokenProb[i]["<UNK>"] = math.log(smoothing/(count+(smoothing*(vocab+1))))
        # self it all
        self.initTagProb = initTagProb
        self.tagToTagProb = tagToTagProb
        self.tagToTokenProb = tagToTokenProb
        

    def most_probable_tags(self, tokens):
        solution = []
        for i in tokens:
            possibleTags = {}
            # Token does not have any tags, use UNK value
            if i not in self.tokenTotalTags:
                solution.append("<UNK>")
            else:
                # Select highest probability out of all tags
                for j in self.tokenTotalTags[i]:
                    possibleTags[j] = self.tagToTokenProb[j][i]
                    currMax = max(possibleTags, key=possibleTags.get)
                solution.append(currMax)
        return solution

    def viterbi_tags(self, tokens):
        T = len(tokens)
        states = self.tagToTagDict.keys()
        viterbi = {}
        backpointer = {}
        # Initialization step
        for i in states:
            if tokens[0] in self.tagToTokenProb[i]:
                viterbi[i] = {1:self.initTagProb[i]+self.tagToTokenProb[i][tokens[0]]}
            else:
                viterbi[i] = {1:self.initTagProb[i]+self.tagToTokenProb[i]["<UNK>"]}
            backpointer[i] = {1:0}
        # Somehow make it work so that UNK->tag doesn't happen
        #   Solution: Let it run using UNK value for tagToToken. However,
        #   when doing tagToTag, use the tag that the UNK value stood in for.
        #   When setting backpointer though, keep track of two values. Actual
        #   backtracing tag, and proper display tag. Note that there is no
        #   difference when not using UNK, but if it is UNK, the display tag
        #   becomes UNK.
        # Iterates through each token
        for i in range(2,T+1):
            # Iterates through each possible current state
            for j in states:
                findingMax = {}
                # Iterates through each possible past state
                for prevState in states:
                    if tokens[i-1] in self.tagToTokenProb[j]:
                        findingMax[prevState] = viterbi[prevState][i-1]+self.tagToTagProb[prevState][j]+self.tagToTokenProb[j][tokens[i-1]]
                    else:
                        findingMax[prevState] = viterbi[prevState][i-1]+self.tagToTagProb[prevState][j]+self.tagToTokenProb[j]["<UNK>"]
                # Found at least one possible tag to transition from
                currMax = max(findingMax, key=findingMax.get)
                viterbi[j][i] = findingMax[currMax]
                # Check if previous tag was UNK
                if tokens[i-2] not in self.tokenTotalTags:
                    isUNK = True
                else:
                    isUNK = False
                # Set backpointer based on previous tag
                if isUNK == False:
                    backpointer[j][i] = (currMax,currMax)
                else:
                    backpointer[j][i] = (currMax,"<UNK>")
        # Termination step
        findBestPath = {}
        # Find best probability among final results
        for i in states:
            findBestPath[i] = viterbi[i][T]
        bestPathPointer = max(findBestPath, key=findBestPath.get)
        bestPathProb = findBestPath[bestPathPointer]
        # Check if last token is has the tag of UNK
        if tokens[T-1] not in self.tokenTotalTags:
            isUNK = True
        else:
            isUNK = False
        # Initialize bestPath based on if last token was UNK
        if isUNK == False:
            bestPath = [bestPathPointer]
        else:
            bestPath = ["<UNK>"]
        t = T
        # Walk backwards through results
        while t > 1:
            # tag pointer
            bestPathPointer = backpointer[bestPathPointer][t][0]
            # tag display
            bestPath = [backpointer[bestPathPointer][t][1]] + bestPath
            t-=1
        return bestPath
