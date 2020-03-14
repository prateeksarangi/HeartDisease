import wfdb

record = wfdb.rdrecord('/Users/ashwini/Downloads/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/training-a/a0001') 
wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015') 
display(record.__dict__)