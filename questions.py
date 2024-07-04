import nltk
import sys
import os
import string
import codecs
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Dictionary for filename + content
    dictionary = dict()
    # Get the path of the directory
    path = os.path.join(directory)
    # Get all filenames in the directory path
    for fileName in os.listdir(path):
        # Get the file path
        filePath = os.path.join(directory, fileName)
        # Open the file from the filepath
        with codecs.open(filePath, encoding='utf-8') as f:
            # Read the file
            fileContent = f.read()
            # Save the files contents to its name in a dictionary
            dictionary[fileName] = fileContent
    # Return the dictionary
    return dictionary


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenize the document
    wordTokens = nltk.word_tokenize(document)
    # Get all stopwords
    stopWords = set(nltk.corpus.stopwords.words('english'))
    # Get all punctuation
    punctuation = string.punctuation
    # Processed words
    words = []
    for word in wordTokens:
        # Check if word is not in stopWords
        if word.lower() not in stopWords:
            # Check if word is not in punctuation
            if word.lower() not in punctuation:
                # If satisfies both statements add to processed words
                words.append(word.lower())
    # Return processed words
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Dictionary for IDF values
    idfValues = dict()
    # Dictionary to keep the count of the words appearing in at least one document
    appeardWords = dict()
    # Loop through all names of documents
    for docName in documents:
        # Get all the words for the document
        docWords = set(documents[docName])
        # Loop through words in document
        for word in docWords:
            # Check if word is in appearedWords and increase its count
            if word in appeardWords:
                appeardWords[word] += 1
            # Check if word appears for 1st time and set count to 1
            else:
                appeardWords[word] = 1
    # Loop through all words in appearedWords
    for word in appeardWords:
        # Calculate the IDF Value for the word
        idfValues[word] = math.log(len(documents) / appeardWords[word])
    # return the dictionary of IDF values
    return idfValues


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Dictionary to store top n files matchin the query
    tfidf = dict()
    # Loop through the names of the files
    for docName in files:
        # Get all words in file
        docWords = files[docName]
        # initialise value of 0 wor each word
        value = 0
        # Loop through words in query
        for word in query:
            # Set the initial idf value to 0
            idfValue = 0
            # Get the count of how many times a words appears in file
            appearedCount = docWords.count(word)
            # If appeared count is greater than 0 get the idf value for the word
            if appearedCount > 0:
                idfValue = idfs[word]
            # Calculate the final tf-idf value for the word
            value += appearedCount * idfValue
        # Save the tf-idf values
        tfidf[docName] = value
    # Sort the dictionary by tf-idf value
    sortedTfidf = sorted(tfidf, key=tfidf.get, reverse=True)
    # Filter the sorted dictionary to return n top files
    sortedTfidf = sortedTfidf[:n]
    # Return sorted dictionary
    return sortedTfidf


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Dictionary to store IDF values
    idf = dict()
    # Dictionary to store the query temr densities
    densities = dict()
    # Loop through all sentences
    for sentence in sentences:
        # Get all sentence words
        sentenceWords = sentences[sentence]
        # Initialise the value and density
        value = 0
        density = 0
        # Loop through words in query
        for word in query:
            # Ignore word if not in the current sentence words
            if word not in sentenceWords:
                continue
            # Set the initial IDF value to 0
            idfValue = 0
            # Get the count of how many times the word appears in a sentence
            appearedCount = sentenceWords.count(word)
            # If the appeared count is more than 0 calculate the IDF value for the word
            if appearedCount > 0:
                idfValue = idfs[word]
            # Sum up the IDF Values
            value += idfValue
            # Caluclate the density
            density += appearedCount / len(sentenceWords)
        # Set the IDF value and the query temr density for the sentence
        idf[sentence] = value
        densities[sentence] = density
    # Sort the dictionary by IDF values
    sortedIdf = sorted(idf, key=lambda x: (idf[x], densities[x]), reverse=True)
    # Filter the sorted dictionary to return n top sentences
    sortedIdf = sortedIdf[:n]
    # Return the sorted dictionary
    return sortedIdf


if __name__ == "__main__":
    main()
