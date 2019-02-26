from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
from bpe import Encoder 
from keras.preprocessing.sequence import pad_sequences

import pandas as pd 
class Data(  snt.AbstractModule ):


	def __init__( self ,  df  , batch_size = 128 , output_size = 3  , name = "data_feeder" ):

		super( Data , self ).__init__( name = name )

		self.batch_size = batch_size
		self.output_size = output_size
		self.encoder = Encoder( 200 , pct_bpe = 0.88 )
		self.df = df
		self.num_samples = df.shape[0]
		self.load_feats( )

	def load_feats( self  ):


		df = self.preparedata( self.df )
		self.encoder.fit( df["Text"].values ) 
		#3log(x+y)
		df.loc[ : , "encoded"] = df["Text"].apply( lambda x: self.encode_feats( x ) )

		labels = df[ ["A-coref" , "B-coref" , "N" ]].values

		seqs = df["encoded"].values 


		seqs_pad = pad_sequences( seqs )

		M = len(seqs_pad[0])
		K1 = len(seqs_pad)
		seqs_pad = seqs_pad.reshape(  ( K1  , M , 1 ))
		seqs_pad = seqs_pad.astype( np.float32 )
		labels = labels.astype( np.float32 )
		self.seqs_pad = seqs_pad 
		self.labels = labels

		self.M = K1 
		return (seqs_pad , labels )

	def preparedata(self , df ):

		df["A-coref"] = df["A-coref"].astype(int)
		df["B-coref"] = df["B-coref"].astype(int )
		df.loc[: , "N"] = 1 - (df["A-coref"] + df["B-coref"])

		df["A"] = df["A"].apply( lambda x: x.lower()  )
		df["B"] = df["B"].apply( lambda x: x.lower() )


		df["Pronoun"] = df["Pronoun"].apply( lambda x: x.lower() )
		df["Text"] = df["Text"].apply( lambda x : x.lower() )

		# removing some things
		df["Text"] = df.apply(lambda x: x["Text"].replace("*" , "x")  , axis = 1   )

		#

		df["Text"] = df.apply(lambda x: self.highlight(x), axis = 1)

		df["Text"] = df.apply(lambda x: self.name_replace( x["Text"] , x["A"] , "subjectone" ) , axis = 1   )
		df["Text"] = df.apply(lambda x: self.name_replace( x["Text"] , x["B"] , "subjecttwo" ) , axis = 1   )

		return df 

	def _build( self , repeat = None  ):

		self.dataset1 = tf.data.Dataset.from_tensor_slices(   self.load_feats()  ).repeat( repeat ).shuffle( self.M ).batch( self.batch_size )
		#data_iterator = dataset1.make_initializable_iterator()
		self.data_iterator = self.dataset1.make_initializable_iterator()
		#feats , labels = self.load_feats()

		#index = np.random.randint( low = 0 , high = len(feats) , size = self.batch_size )

		x , y =  self.data_iterator.get_next()
		#x = feats[ index, : ]
		#y = labels[ index , : ]


		#x = x.reshape( ( len()  , self.batch_size , 1 ))
		print("SI no me conoces no me se;ales ")
		print( x.shape )
		print(y.shape )
		return x , y 


	def encode_feats( self , x ):
	


		txt_encoded = list(next( self.encoder.transform( [ x ])  ))
		return np.array( txt_encoded ) 

	def get_labels( self , x ):

		A_ref = x["A-coref"]
		B_ref = x["B-coref"]
		label = [ A_ref , B_ref ]
		if A_ref == False and B_ref == False:

			label.append( True )

			return( np.array( label ).reshape(3) )
		else:
			label.append( False )
			
			return( np.array( label ).reshape(3) )


	def highlight( self ,   x ):
	    
	    # x is a row
		offset_pronoun = x["Pronoun-offset"] 
	    
		pronoun = x["Pronoun"]
		len_pronoun = len( pronoun )
	    #print( len_pronoun )
		text = x["Text"]
	    
		new_text = text[: offset_pronoun] + "*" + pronoun +"*" + text[  offset_pronoun+len_pronoun:  ]
	    
		return new_text

	def name_replace(self , s, r1, r2):
		s = str(s).replace(r1,r2)
		return s

	def loss( self , y_true , y_pred):

		#return tf.nn.sigmoid_cross_entropy_with_logits( labels = y_true , logits = y_pred )
		return tf.losses.log_loss( labels = y_true , predictions = y_pred )