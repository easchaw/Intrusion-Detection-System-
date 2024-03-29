# Sequential Anomaly Detection

This work describes a computational efficient anomaly based intrusion detection system based on Recurrent Neural Networks.
Using Gated Recurrent Units rather than the normal LSTM networks it is possible to obtain a set of comparable results with reduced training times. 
The incorporation of stacked CNNs with GRUs leads to improved anomaly IDS. Intrusion Detection is based on determining the probability of a particular call sequence occurring from a language model trained on normal call sequences from the ADFA Data set of system call traces . 
Sequences with a low probability of occurring are classified as an anomaly.

![methodology](https://user-images.githubusercontent.com/20104666/157997407-a0a2e21e-cba5-4f20-89eb-2422a24427af.JPG)

