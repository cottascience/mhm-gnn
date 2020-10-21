# MHM-GNN

![](diagram.png)

## Setting up

To compile the C++ sampling code, you should have conda and g++ compiler (version 5.4+) installed. We have tested the code on Ubuntu servers, thus we recommend its use, although it should work fine in other unix-based systems.

```
cd external_libs
unzip rgpm.zip
cd rgpm
./compile.sh
```

## Downloading Data
Please, run the following commands to download and set up the data folder.

```
wget https://www.dropbox.com/s/24037etsfe9iwub/mhm-data.zip
mv mhm-data data
cd data
unzip cora.zip
unzip citeseer.zip
unzip pubmed.zip
unzip dblp.zip
unzip steam.zip
unzip renttherunway.zip
```

## Running Experiments

Make sure you have Python 3.x and the latest versions of Pytorch and Pytorch Geometric installed.

You can run the experiments from Tables 1,2,3 from the original paper with the scripts [dataset]-[k].py
