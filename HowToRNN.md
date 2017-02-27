There is a bit of controversy for what concerns RNNs. 
For a long time people only used [LSM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976602760407955)
and [ESN](http://minds.jacobs-university.de/sites/default/files/uploads/papers/EchoStatesTechRep.pdf)
as a general framework for RNNs, problem is you don't really train these network
but only their input and output weights, the rest is kind of random.

After 2009
the model introduced by Schimdhuber in 1997 [LSTM](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
started showing overall better performance so everyone started using it and LSM and ESN
were not used anymore. For what concerns studying RNNs behavior they are still good models, but if you want to do Machine Learning LSTM is definitely the best solution.

There is one modification to the original LSTM that has been done by Schimdhuber later,
LSTM is now used with these modification, even though people think that it has
always been like that, [peepholes](http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf)
did not exist in the first version of LSTM :)

Now, in the beginning LSTM was used for sequence labelling and classification (and it is mostly used like this now), but Schmidhuber also did some work on [unsupervised methods](ftp://ftp.idsia.ch/pub/juergen/icann2001unsup.pdf) to train LSTM, I don't know if you need it, but is cool to know!

Another add-on that people started using was [Bidirectional RNNs](http://dl.acm.org/citation.cfm?id=2205129) which were actually introduced long before people started using LSTM for everything!

Then there are some papers were Graves started destroying a bunch of benchmarks with LSTM!
Sequence Labelling is basically his [PhD Thesis](https://www.cs.toronto.edu/~graves/preprint.pdf)  
[Sequence generation](https://arxiv.org/abs/1308.0850) was actually pretty cool, you can play around with their [demo](http://www.cs.toronto.edu/~graves/handwriting.html).
Graves also started doing [speech recognition](https://www.cs.toronto.edu/~fritz/absps/RNN13.pdf) of course.

And he introduced also a cool algorithm, which is widely used right now to get better perfomance in speech recognition but can also be used in general for sequence labelling.
it's called [CTC](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf) basically if you have two sequence like speech and its transcription it allows you to align the two sequences.

Next is the introduction of [GRU](https://arxiv.org/abs/1406.1078) which is basically a simpler version of LSTM (less parameters), but that works with the same principles. People showed that the two models are exactly the same if you put a bias of 1 in the forget gate of LSTM. In this paper they also introduced encoder-decoder architecture, which is the state of the art architecture for NLP. Basically again is a way of mapping two sequence that lie in different spaces and have different lengths.

What came out after encoder-decoder was the [attention mechanism](https://arxiv.org/abs/1409.0473), again pretty cool for NLP but also in general for sequence labelling and so on.

[This](https://arxiv.org/abs/1411.4389) is a nice example of using combination of LSTM and CNN, which is a pretty powerful combination.
Recently DeepMind has started looking at [image processing](https://arxiv.org/abs/1601.06759) with LSTM without CNNs, it's interesting as well .

It's a lot of stuff and probably you don't need to read everything. But at least here you have a somewhat complete story. I would suggest Graves Thesis as a general overview of the RNNs framework! Then the rest is basically nice adds-on and tricks. I have more stuff that you can read if you want! But let me know what you think is more useful for you!

Cheers,
+Enea
