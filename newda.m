[s1  fs] = audioread('factory.wav');
s1 = s1(1:166000,1);
audiowrite('new.wav',s1,fs);