def words2characters(words):
    
    characters = []
    
    for word in words:
        for char in str(word):
            characters.append(char)
    
    return characters
