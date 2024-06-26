# Example A
Let's quickly review the data we'll be using to predict house prices using the describe method and the head method, which shows the top few rows:

    X.describe()

    Rooms	Bathroom	Landsize	Lattitude	Longtitude
    count	6196.000000	6196.000000	6196.000000	6196.000000	6196.000000
    mean	2.931407	1.576340	471.006940	-37.807904	144.990201
    std	        0.971079	0.711362	897.449881	0.075850	0.099165
    min	        1.000000	1.000000	0.000000	-38.164920	144.542370
    25%	        2.000000	1.000000	152.000000	-37.855438	144.926198
    50%	        3.000000	1.000000	373.000000	-37.802250	144.995800
    75%	        4.000000	2.000000	628.000000	-37.758200	145.052700
    max	        8.000000	8.000000	37000.000000	-37.457090	145.526350


    X.head()

	Rooms	Bathroom	Landsize	Lattitude	Longtitude
    1	2	1.0	156.0	-37.8079	144.9934
    2	3	2.0	134.0	-37.8093	144.9944
    4	4	1.0	120.0	-37.8072	144.9941
    6	3	2.0	245.0	-37.8024	144.9993
    7	2	1.0	256.0	-37.8060	144.9954


Visually checking your data with these commands is an important part of a data scientist's job. You'll frequently find surprises in the dataset that deserve further inspection.


# Example B

- Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good
practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose. We now have a fitted model that we can use to make predictions.
- In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first few rows
of the training data to see how the predict function works.

      print("Making predictions for the following 5 houses:")
      print(X.head())
      print("The predictions are")
      print(melbourne_model.predict(X.head()))

      Making predictions for the following 5 houses:
             Rooms  Bathroom  Landsize  Lattitude  Longtitude
      1      2       1.0     156.0   -37.8079    144.9934
      2      3       2.0     134.0   -37.8093    144.9944
      4      4       1.0     120.0   -37.8072    144.9941
      6      3       2.0     245.0   -37.8024    144.9993
      7      2       1.0     256.0   -37.8060    144.9954

      The predictions are
      [1035000. 1465000. 1600000. 1876000. 1636000.]

