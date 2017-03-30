import csvTutorial

# csvTutorial.printRows('train.csv',10,15)

data = csvTutorial.readCSVfile('train.csv')

print(csvTutorial.baseLine(data[1:], 5))
