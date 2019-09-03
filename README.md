# Semantic Scholar Sampler

Sample abstracts from a semantic scholar gzip dataset files.

## Configuration parameters ##

The configuration file (/src/main/resources/config.properties) points to the input data and other parameters. 


```
max.abstracts.sample=500
text.input.venues=src/main/resources/venues.txt
output.filename=test.json
gzip.input.folder=src/main/resources/
minimum.year=2018
```

#### Input data 
The input data files are GZIP compressed and can be downloaded from the [Semantic Scholar Open Research Corpus](https://api.semanticscholar.org/corpus/download/). Download them and specify the path in the config.properties file. 

#### Sample abstracts

In order to obtain a sample of abstracts, please follow the steps:

- **Clone** this project in your computer
- **Enter** in the project directory and execute:
```
mvn clean package
```
- You should see this ".jar" file in your /target directory: 

```
semanticscholarindex-0.0.1-SNAPSHOT-jar-with-dependencies.jar
```

- **Execute** the jar file:

```
java -jar EventKG.graphGenerator-0.0.1-SNAPSHOT-jar-with-dependencies.jar
```
- **Wait** until the program concludes the execution
- **Output** is a JSON file (one abstract per line)
 

#### License ####

This project is licensed under the terms of the MIT license (see LICENSE.txt).

