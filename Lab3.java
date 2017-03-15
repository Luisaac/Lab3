/**
 * @Author: Yuting Liu and Jude Shavlik.  
 * 
 * Copyright 2017.  Free for educational and basic-research use.
 * 
 * The main class for Lab3 of cs638/838.
 * 
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * 
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 * 
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import javax.imageio.ImageIO;

public class Lab3 {

	public static int     imageSize = 32; // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).  
	// You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
	// ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
	private static enum    Category { airplanes, butterfly, flower, grand_piano, starfish, watch };  // We'll hardwire these in, but more robust code would not do so.

	private static final Boolean    useRGB = true; // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
	private static       int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.

	final private static String    modelToUse = "deep"; // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
	private static int       inputVectorSize;         // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.  
	// Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.  
	// The last element in this vector holds the 'teacher-provided' label of the example.

	//final private static double eta       =    0.1, fractionOfTrainingToUse = 1.00, dropoutRate = 0.50; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).
	final private static int    maxEpochs = 5; // Feel free to set to a different value.



	final public static double learningRate = 0.01;
	final public static double momentum = 0.01;
	final public static double parameter = 0.03;
	final public static int numHU = 100;
	final public static int numOut = 6;
	final static int kernal_length1 = 5;
	final static int pooling_length1 = 2;
	final static int batchsize = 10;

	//TODO num_plate = 20;
	final static int num_plate = 20;
	final static int secondLayerSize = (imageSize-kernal_length1+1)/pooling_length1;
	final static int numLinks = (secondLayerSize-kernal_length1+1)/pooling_length1 * ((secondLayerSize-kernal_length1+1)/pooling_length1)*num_plate;
	static Layer C1_layer = null;
	static Layer C2_layer = null;
	static FCLayer hiddenLayer;
	static FCLayer outLayer;

	// bacth: store average delta
	static double[][] hiddenDelta = new double[numHU][numLinks+1];
	static double[][] outDelta = new double[numOut][numHU+1];
	static Vector<double[][][]> conv1Delta = new Vector<double[][][]>(num_plate);
	static Vector<double[][][]> conv2Delta = new Vector<double[][][]>(num_plate);
	static double[] conv1Bias = new double[num_plate];
	static double[] conv2Bias = new double[num_plate];

	static Vector<double[][]> output_layer = new Vector<double[][]>();
	static Vector<double[][]> deltas_2 = null;
	static Vector<double[][]> deltas_1 = null;
	static boolean createExtraTrainingExamples = false;
	private static PrintWriter out;

	static double epsilon = Math.pow(10, -5);


	public static void main(String[] args) {
		String trainDirectory = "images/trainset/";
		String  tuneDirectory = "images/tuneset/";
		String  testDirectory = "images/testset/";

		if(args.length > 5) {
			System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
			System.exit(1);
		}
		if (args.length >= 1) { trainDirectory = args[0]; }
		if (args.length >= 2) {  tuneDirectory = args[1]; }
		if (args.length >= 3) {  testDirectory = args[2]; }
		if (args.length >= 4) {  imageSize     = Integer.parseInt(args[3]); }

		// Here are statements with the absolute path to open images folder
		File trainsetDir = new File(trainDirectory);
		File tunesetDir  = new File( tuneDirectory);
		File testsetDir  = new File( testDirectory);

		// create three datasets
		Dataset trainset = new Dataset();
		Dataset  tuneset = new Dataset();
		Dataset  testset = new Dataset();

		// Load in images into datasets.
		long start = System.currentTimeMillis();
		loadDataset(trainset, trainsetDir);
		System.out.println("The trainset contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		////////////////////
		if (createExtraTrainingExamples) {
			start = System.currentTimeMillis();
			Dataset trainsetExtras = new Dataset();

			// Flipping watches will mess up the digits on the watch faces, but that probably is ok.
			for (Instance origTrainImage : trainset.getImages()) {
				if (!"airplanes".equals(  origTrainImage.getLabel()) &&  // Airplanes all 'face' right and up, so don't flip left-to-right or top-to-bottom.
						!"grand_piano".equals(origTrainImage.getLabel())) {  // Ditto for pianos.

					trainsetExtras.add(origTrainImage.flipImageLeftToRight());

					if (!"butterfly".equals(origTrainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't flip top-to-bottom.
							!"flower".equals(   origTrainImage.getLabel()) &&  // Ditto for flowers.
							!"starfish".equals( origTrainImage.getLabel())) {  // Star fish are standardized to 'point up.
						trainsetExtras.add(origTrainImage.flipImageTopToBottom());
					}
				}

				for (int shiftCopies = 1; shiftCopies <= 2; shiftCopies++) {
					trainsetExtras.add(origTrainImage.shiftImage(shiftCopies, shiftCopies));
				}
			}


			int[] countOfCreatedTrainingImages = new int[Category.values().length];
			for (Instance createdTrainImage : trainsetExtras.getImages()) {
				// Keep more of the less common categories?
				double probOfKeeping = 1.0;

				// Trainset counts: airplanes=127, butterfly=55, flower=114, piano=61, starfish=51, watch=146
				if      ("airplanes".equals(  createdTrainImage.getLabel())) probOfKeeping = 0.50; // Only shifted, so fewer created.
				else if ("butterfly".equals(  createdTrainImage.getLabel())) probOfKeeping = 1.00; // No top-bottom flips, so fewer created.
				else if ("flower".equals(     createdTrainImage.getLabel())) probOfKeeping = 0.33; // No top-bottom flips, so fewer created.
				else if ("grand_piano".equals(createdTrainImage.getLabel())) probOfKeeping = 1.00; // Only shifted, so fewer created.
				else if ("starfish".equals(   createdTrainImage.getLabel())) probOfKeeping = 1.00; // No top-bottom flips, so fewer created.
				else if ("watch".equals(      createdTrainImage.getLabel())) probOfKeeping = 0.20; // Already have a lot of these.
				//       else waitForEnter("Unknown label: " + createdTrainImage.getLabel());

				if (random() <= probOfKeeping) {
					countOfCreatedTrainingImages[convertCategoryStringToEnum(createdTrainImage.getLabel()).ordinal()]++;
					trainset.add(createdTrainImage);
				}
			}
			for (Category cat : Category.values()) {
				System.out.println(" Created" + padLeft(comma(countOfCreatedTrainingImages[cat.ordinal()]), 4) + " 'tweaked' images of " + cat + ".");
			}
			System.out.println("Created " + comma(trainsetExtras.getSize()) + " new training examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
		}
		/////////////////////

		start = System.currentTimeMillis();
		loadDataset(tuneset, tunesetDir);
		System.out.println("The  testset contains " + comma( tuneset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		loadDataset(testset, testsetDir);
		System.out.println("The  tuneset contains " + comma( testset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");


		// Now train a Deep ANN.  You might wish to first use your Lab 2 code here and see how one layer of HUs does.  Maybe even try your perceptron code.
		// We are providing code that converts images to feature vectors.  Feel free to discard or modify.
		start = System.currentTimeMillis();
		trainANN(trainset, tuneset, testset);


		System.out.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");

	}

	public static void loadDataset(Dataset dataset, File dir) {
		for(File file : dir.listFiles()) {
			// check all files
			if(!file.isFile() || !file.getName().endsWith(".jpg")) {
				continue;
			}
			//String path = file.getAbsolutePath();
			BufferedImage img = null, scaledBI = null;
			try {
				// load in all images
				img = ImageIO.read(file);
				// every image's name is in such format:
				// label_image_XXXX(4 digits) though this code could handle more than 4 digits.
				String name = file.getName();
				int locationOfUnderscoreImage = name.indexOf("_image");

				// Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
				if (imageSize != 128) {
					scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
					Graphics2D g = scaledBI.createGraphics();
					g.drawImage(img, 0, 0, imageSize, imageSize, null);
					g.dispose();
				}

				Instance instance = new Instance(scaledBI == null ? img : scaledBI, name,name.substring(0, locationOfUnderscoreImage));

				dataset.add(instance);
			} catch (IOException e) {
				System.err.println("Error: cannot load in the image file");
				System.exit(1);
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////

	private static Category convertCategoryStringToEnum(String name) {
		if ("airplanes".equals(name))   return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
		if ("butterfly".equals(name))   return Category.butterfly;
		if ("flower".equals(name))      return Category.flower;
		if ("grand_piano".equals(name)) return Category.grand_piano;
		if ("starfish".equals(name))    return Category.starfish;
		if ("watch".equals(name))       return Category.watch;
		throw new Error("Unknown category: " + name);		
	}

	public static double getRandomWeight(int fanin, int fanout) { // This is one 'rule of thumb' for initializing weights.  Fine for perceptrons and one-layer ANN at least.
		double range = Math.max(Double.MIN_VALUE, 1.0 / Math.sqrt(fanin+fanout));
		return (2.0 * random() - 1.0) * range;
		//return -0.03+0.06*random();
	}

	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature vector.
	private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) { // If only using GREY, then offset = 0;  Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
		return ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I have not used this, so might need debugging.
	}

	///////////////////////////////////////////////////////////////////////////////////////////////


	// Return the count of TESTSET errors for the chosen model.
	private static int trainANN(Dataset trainset, Dataset tuneset, Dataset testset) {
		Instance sampleImage = trainset.getImages().get(0); // Assume there is at least one train image!
		inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1; // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it).  The final 1 is for the CATEGORY.

		// For RGB, we use FOUR input units per pixel: red, green, blue, plus grey.  Otherwise we only use GREY scale.
		// Pixel values are integers in [0,255], which we convert to a double in [0.0, 1.0].
		// The last item in a feature vector is the CATEGORY, encoded as a double in 0 to the size on the Category enum.
		// We do not explicitly store the '-1' that is used for the bias.  Instead code (to be written) will need to implicitly handle that extra feature.
		System.out.println("\nThe input vector size is " + comma(inputVectorSize - 1) + ".\n");

		Vector<Vector<double[][]>> trainFeatureVectors = new Vector<Vector<double[][]>>(trainset.getSize());
		Vector<Vector<double[][]>>  tuneFeatureVectors = new Vector<Vector<double[][]>>( tuneset.getSize());
		Vector<Vector<double[][]>>  testFeatureVectors = new Vector<Vector<double[][]>>( testset.getSize());

		Vector<Double> trainLabels = new Vector<Double>(trainset.getSize());
		Vector<Double> tuneLabels = new Vector<Double>(tuneset.getSize());
		Vector<Double> testLabels = new Vector<Double>(testset.getSize());

		C1_layer = new Layer(4,20,kernal_length1,pooling_length1,imageSize);
		C2_layer = new Layer(20,20,kernal_length1,pooling_length1,secondLayerSize);
		deltas_2 = new Vector<double[][]>(C2_layer.plates.length);
		deltas_1 = new Vector<double[][]>(C1_layer.plates.length);
		hiddenLayer = new FCLayer(numHU,numLinks+1, 0);
		outLayer = new FCLayer(numOut,numHU+1,1);
		long start = System.currentTimeMillis();
		fillFeatureVectors(trainFeatureVectors, trainset, trainLabels);
		System.out.println("Converted " + trainFeatureVectors.size() + " TRAIN examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		fillFeatureVectors( tuneFeatureVectors,  tuneset, tuneLabels);
		System.out.println("Converted " +  tuneFeatureVectors.size() + " TUNE  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		fillFeatureVectors( testFeatureVectors,  testset, testLabels);
		System.out.println("Converted " +  testFeatureVectors.size() + " TEST  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		System.out.println("\nTime to start learning!");

		//TODO mark 
		try {
			out = new PrintWriter(new BufferedWriter(new FileWriter("output.out")));
		} catch (IOException e) {
			e.printStackTrace();
		}
		//		C1_layer = new Layer(4,num_plate,kernal_length1,pooling_length1,16);
		//		C2_layer = new Layer(num_plate,num_plate,kernal_length1,pooling_length1,7);
		//		deltas_2 = new Vector<double[][]>(C2_layer.plates.length);
		//		deltas_1 = new Vector<double[][]>(C1_layer.plates.length);
		//		hiddenLayer = new FCLayer(numHU,numLinks+1, 0);
		//		outLayer = new FCLayer(numOut,numHU+1,1);
		//		trainLabels = new Vector<Double>();
		//		for(int i = 0; i < 4;i++){
		//			Vector<double[][]> ins = new Vector<double[][]>();
		//			for(int m = 0; m < 4;m++){
		//				double[][] tmp = new double[16][16];
		//				for(int j = 0; j < 16;j++){
		//					for(int k = 0; k< 16; k++){
		//						tmp[j][k] = 1;
		//					}
		//				}
		//				ins.add(tmp);
		//			}
		//			trainLabels.addElement((double)i);
		//			trainFeatureVectors.add(ins);
		//		}
		//
		//		for(int i = 0; i < 4;i++){
		//			Vector<double[][]> ins = new Vector<double[][]>();
		//			for(int m = 0; m < 4;m++){
		//				double[][] tmp = new double[16][16];
		//				for(int j = 0; j < 16;j++){
		//					for(int k = 0; k< 16; k++){
		//						tmp[j][k] = 0.1;
		//					}
		//				}
		//				ins.add(tmp);
		//			}
		//			testFeatureVectors.add(ins);
		//		}

		// Call your Deep ANN here.  We recommend you create a separate class file for that during testing and debugging, but before submitting your code cut-and-paste that code here.

		//		if      ("perceptrons".equals(modelToUse)) return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Either comment out this line or just right a 'dummy' function.
		//		else if ("oneLayer".equals(   modelToUse)) return trainOneHU(      trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Ditto.
		//		else 
		if ("deep".equals(       modelToUse)){
			//for(int i = 0; i < maxEpochs; i++){
			for(int i = 0; i <40; i++){
				permute(trainFeatureVectors, trainLabels);
				System.out.println("epoch: " +i);
				System.out.println("Train Accuracy at epoch " +i);


				Vector<Vector<double[][]>> batch = new Vector<Vector<double[][]>>(batchsize);
				Vector<Double> labels = new Vector<Double>(batchsize);
				for(int index = trainFeatureVectors.size();index < trainFeatureVectors.size()-batchsize;index++){
					int j = random0toNminus1(index);
					Vector<double[][]> tmp = trainFeatureVectors.get(j);
					labels.add(trainLabels.get(j));
					batch.add(tmp);

				}
				trainDeep(batch, 0, labels);
				System.out.println("Tune Accuracy at epoch " +i);
				trainDeep(tuneFeatureVectors, 1, tuneLabels);
				System.out.println("Test Accuracy at epoch " +i);
				trainDeep(testFeatureVectors, 1, testLabels);
			}
		}
		out.close();
		return -1;
	}

	private static void fillFeatureVectors(Vector<Vector<double[][]>> featureVectors, Dataset dataset, Vector<Double> imageLabels) {

		for (Instance image : dataset.getImages()) {
			double label = (double) convertCategoryStringToEnum(image.getLabel()).ordinal();
			imageLabels.add(label);
			Vector<double[][]> result = new Vector<double[][]>();
			int len = image.getRedChannel().length;
			double[][] red = new double[len][len];
			double[][] green = new double[len][len];
			double[][] blue = new double[len][len];
			double[][] gray = new double[len][len];
			for(int i = 0; i<len;i++){
				for(int j = 0;j<len;j++){
					gray[i][j] = image.getGrayImage()[i][j]/255.0;
					if(useRGB){
						red[i][j] = image.getRedChannel()  [i][j] / 255.0;
						green[i][j] = image.getGreenChannel()[i][j]/255.0;
						blue[i][j] = image.getBlueChannel()[i][j]/255.0;
						//System.out.println(red[i][j] + " "+green[i][j] +" "+blue[i][j]);
					}
				}
			}
			if(useRGB){
				result.add(red);
				result.add(blue);
				result.add(green);
			}
			result.add(gray);
			featureVectors.addElement(result);
		}
	}


	////////////////////  Some utility methods (cut-and-pasted from JWS' Utils.java file). ///////////////////////////////////////////////////

	private static final long millisecInMinute = 60000;
	private static final long millisecInHour   = 60 * millisecInMinute;
	private static final long millisecInDay    = 24 * millisecInHour;
	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}
	public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
		if (millisec ==    0) { return "0 seconds"; } // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec <  1000) { return comma(millisec) + " milliseconds"; } // Or just comment out these two lines?
		if (millisec > millisecInDay)    { return comma(millisec / millisecInDay)    + " days and "    + convertMillisecondsToTimeSpan(millisec % millisecInDay,    digits); }
		if (millisec > millisecInHour)   { return comma(millisec / millisecInHour)   + " hours and "   + convertMillisecondsToTimeSpan(millisec % millisecInHour,   digits); }
		if (millisec > millisecInMinute) { return comma(millisec / millisecInMinute) + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits); }

		return truncate(millisec / 1000.0, digits) + " seconds"; 
	}

	public static String comma(int value) { // Always use separators (e.g., "100,000").
		return String.format("%,d", value);    	
	}    
	public static String comma(long value) { // Always use separators (e.g., "100,000").
		return String.format("%,d", value);    	
	}   
	public static String comma(double value) { // Always use separators (e.g., "100,000").
		return String.format("%,f", value);    	
	}
	public static String padLeft(String value, int width) {
		String spec = "%" + width + "s";
		return String.format(spec, value);    	
	}

	/**
	 * Format the given floating point number by truncating it to the specified
	 * number of decimal places.
	 * 
	 * @param d
	 *            A number.
	 * @param decimals
	 *            How many decimal places the number should have when displayed.
	 * @return A string containing the given number formatted to the specified
	 *         number of decimal places.
	 */
	public static String truncate(double d, int decimals) {
		double abs = Math.abs(d);
		if (abs > 1e13)             { 
			return String.format("%."  + (decimals + 4) + "g", d);
		} else if (abs > 0 && abs < Math.pow(10, -decimals))  { 
			return String.format("%."  +  decimals      + "g", d);
		}
		return     String.format("%,." +  decimals      + "f", d);
	}

	/** Randomly permute vector in place.
	 *
	 * @param <T>  Type of vector to permute.
	 * @param vector Vector to permute in place. 
	 */
	public static void permute(Vector<Vector<double[][]>> vector, Vector<Double> labels) {
		if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an unbiased permute; I prefer (1) assigning random number to each element, (2) sorting, (3) removing random numbers.
			// But also see "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which justifies this.
			/*	To shuffle an array a of n elements (indices 0..n-1):
 									for i from n - 1 downto 1 do
      								j <- random integer with 0 <= j <= i
      								exchange a[j] and a[i]
			 */

			for (int i = vector.size() - 1; i >= 1; i--) {  // Note from JWS (2/2/12): to match the above I reversed the FOR loop that Trevor wrote, though I don't think it matters.
				int j = random0toNminus1(i + 1);
				if (j != i) {
					Vector<double[][]> swap =    vector.get(i);
					vector.set(i, vector.get(j));
					vector.set(j, swap);

					double swap2 = labels.get(i);
					labels.set(i, labels.get(j));
					labels.set(j, swap2);
				}
			}
		}
	}

	public static Random randomInstance = new Random(638*838);  // Change the 638 * 838 to get a different sequence of random numbers.

	/**
	 * @return The next random double.
	 */
	public static double random() {
		return randomInstance.nextDouble();
	}

	/**
	 * @param lower
	 *            The lower end of the interval.
	 * @param upper
	 *            The upper end of the interval. It is not possible for the
	 *            returned random number to equal this number.
	 * @return Returns a random integer in the given interval [lower, upper).
	 */
	public static int randomInInterval(int lower, int upper) {
		return lower + (int) Math.floor(random() * (upper - lower));
	}


	/**
	 * @param upper
	 *            The upper bound on the interval.
	 * @return A random number in the interval [0, upper).
	 * @see Utils#randomInInterval(int, int)
	 */
	public static int random0toNminus1(int upper) {
		return randomInInterval(0, upper);
	}

	private static int trainDeep(Vector<Vector<double[][]>> trainFeatureVectors, int type, Vector<Double> imageLabels) {



		double[][] confusion = new double[6][6];
		int correctLabels = 0;

		for(int i = 0; i < trainFeatureVectors.size(); i++){
			//for(int i = 0; i < 1; i++){
			//	System.out.println(i);
			//get labels
			int label = imageLabels.get(i).intValue();
			//double[] targets = getLabels(label);

			// forward
			Vector<double[][]> v = trainFeatureVectors.get(i);
			//double[][] input = transform(trainFeatureVectors.get(i));
			//v.add(input);

		

			//edit 1
			//System.out.println("Convolutional layer 1: ");
			Vector<double[][]> temp = C1_layer.feedForward(v);
			//									System.out.println(temp.get(0)[0][0]);
			//									System.out.println(temp.get(5)[0][0]);
			//									System.out.println(temp.get(10)[0][0]);
			//			System.out.println("Convolutional layer 2: ");
			output_layer =  C2_layer.feedForward(temp);
			//															System.out.println(output_layer.get(0)[0][0]);
			//															System.out.println(output_layer.get(5)[0][0]);
			//															System.out.println(output_layer.get(10)[0][0]);
			double[] inputsFCLayer = C2_layer.output1D();

			double[] hiddenOut = hiddenLayer.feedForward(inputsFCLayer);
			//									System.out.println(hiddenOut[0]);
			//									System.out.println(hiddenOut[150]);
			//									System.out.println(hiddenOut[299]);
			double[] outLayer_outputs = outLayer.feedForward(hiddenOut);

		
			//for(int ia = 0; ia < outLayer_outputs.length; ia++) System.out.println(outLayer_outputs[ia]);

			// backward only for train
			if(type == 0){
				backward(inputsFCLayer,v,label);	
			}

			int predicted = outLayer.predicted;
			if (outLayer.predicted == label){
				correctLabels++;
			}
		
			//if(predicted == label) out.println("Predict: "+ predicted +" Correct: " + label + " GOOD!");
			//else out.println("Predict: "+ predicted +" Correct: " + label);

			confusion[(int)label][predicted]++;

			C2_layer.clearMax();
			C1_layer.clearMax();
			//out.println("image: "+ i);
		}
		//clear delta,
		//TODO update weights and clear delta
		hiddenDelta = 0.0;
		outDelta = 0.0;
		conv1Delta = 0.0;
		conv2Delta = 0.0;
		
		for(int i = 0;i<numOut;i++){
			outLayer.getNeurons(i).updateWeights(outDelta[i]);
		}



		System.out.println("Accuracy = " + correctLabels/(double)trainFeatureVectors.size());

		for(int i = 0; i < 6; i++){
			for(int j = 0; j < 6 ;j++){
				System.out.print(confusion[i][j]+"\t");
			}
			System.out.println();
		}

		return -1;
	}

	public static void backward(double[] inputs,Vector<double[][]> v, int targets){
		double[] hiddenOut = hiddenLayer.getOutputs();
		double[] finalOut = outLayer.getOutputs();
		//backpropagation
		double[] errorWRTOutput = new double[numOut];
		double[] errorWRTHiddenOut = new double[numHU];
		for(int i = 0;i<numOut;i++){
			errorWRTOutput[i] = finalOut[i] - ((targets == i)?1:0);
			//out.println("outNueron: "+i+", Outerror: "+errorWRTOutput[i]+ ", output:"+finalOut[i]);
		}


		for (int i = 0; i < numHU; i++) {
			// derivatives of error wrt output FCLayers
			double tmp = 0.0;
			for (int j = 0; j < numOut; j++) {
				tmp += errorWRTOutput[j] * outLayer.getNeurons(j).getWeight(i);
			}

			errorWRTHiddenOut[i] = tmp * hiddenLayer.getNeurons(i).pdOutputWRTNetout();	
			//out.println("hiddenNueron: "+i+", Hiddenerror: "+errorWRTHiddenOut[i] + ", pd:"+hiddenLayer.getNeurons(i).pdOutputWRTNetout() +", hiddenWeight:" +hiddenLayer.getNeurons(i).getWeight(0));
		}


		Vector<double[][]> errorWRTInput = new Vector<double[][]>();
		int len = C2_layer.plates[0].matrix2.length;
		for(int k = 0; k<C2_layer.num_plate;k++){
			double[][] delta = new double[len][len];
			for(int i =0;i<len;i++){
				for(int m =0;m<len;m++){
					double tmp = 0.0;
					for(int j = 0;j<numHU;j++){
						tmp += errorWRTHiddenOut[j] * hiddenLayer.getNeurons(j).getWeight(m+i*len+len*len*k);
					}
					delta[i][m] = tmp;
				}
			}
			errorWRTInput.add(delta);
		}


		int c2_maxtrix1_len = C2_layer.plates[0].matrix1.length;
		int c1_maxtrix1_len = C1_layer.plates[0].matrix1.length;
		int c1_marix2_len = C1_layer.plates[0].matrix2.length;
		deltas_2.clear();
		deltas_1.clear();
		// step 1
		// delta 2
		for(int i = 0; i < C2_layer.plates.length; i++){
			//System.out.println("\n"+i+" layer count delta:");
			double [][] local_delta2 = new double [c2_maxtrix1_len][c2_maxtrix1_len];
			for(int j = 0; j < c2_maxtrix1_len;j++){
				for(int k = 0; k < c2_maxtrix1_len;k++){
					if(C2_layer.plates[i].useAsMax[j][k] == true){
						local_delta2[j][k] = ((C2_layer.plates[i].inactivated[j][k]>0)?1:0.01)*errorWRTInput.get(i)[j/C2_layer.pooling_length][k/C2_layer.pooling_length];
					}
				}
			}
			deltas_2.add(local_delta2);
		}


		// step 2
		// delta 1
		for(int i = 0; i < C1_layer.plates.length; i++){
			//		System.out.println("\n"+i+" layer count delta:");
			double [][] local_delta1 = new double [c1_maxtrix1_len][c1_maxtrix1_len];
			double[][] mhs_matrix = new double[c1_marix2_len][c1_marix2_len];

			for(int C2_index = 0; C2_index < C2_layer.plates.length; C2_index++){
				for(int j = 0; j < c2_maxtrix1_len; j++){
					for(int k = 0 ; k < c2_maxtrix1_len; k++){
						for(int ki = 0; ki < C2_layer.kernal_length; ki++){
							for(int kj = 0; kj < C2_layer.kernal_length; kj++){
								mhs_matrix[j+ki][k+kj] +=  deltas_2.get(C2_index)[j][k] * C2_layer.kernals.get(C2_index)[i][ki][kj];
							}
						}

					}
				}

			}

			for(int j = 0;j<c1_maxtrix1_len;j++){
				for(int k = 0; k< c1_maxtrix1_len;k++){
					if(C1_layer.plates[i].useAsMax[j][k] == true){
						local_delta1[j][k] = mhs_matrix[j/C1_layer.pooling_length][k/C1_layer.pooling_length] * ((C1_layer.plates[i].inactivated[j][k]>0)?1:0.01);
						//System.out.println("delta1: "+local_delta1[j][k]);
					}
				}
			}
			deltas_1.add(local_delta1);
			//System.out.println("bbbbbb: " + deltas_1.get(i)[0][0] + " local delta" + local_delta1[0][0]);
		}


		//update output weights
		for(int i = 0;i<numOut;i++){
			//errorWRTOutput[i] = outLayer.getNeurons(i).pdErrorWRTNetout(targets);
			double[] UpdatedOutWeights = new double[numHU+1];
			for (int j = 0; j < numHU+1; j++) {
				double errorWRTweight;
				if(j == numHU){
					errorWRTweight = errorWRTOutput[i]* outLayer.getNeurons(i).bias;
				}else{
					errorWRTweight = errorWRTOutput[i] * hiddenOut[j];
				}
				// get previous change in weight
				errorWRTweight /= inputs.length;
				errorWRTweight += 0.5*outLayer.getNeurons(i).getWeight(j);
				
				//TODO no need to uddate weights here, save average and add momentum and ADAM
				double change = errorWRTweight*learningRate +momentum * outLayer.getNeurons(i).getV(j);
				UpdatedOutWeights[j] = outLayer.getNeurons(i).getWeight(j) - change;
				outLayer.getNeurons(i).setV(change, j);
				//				
				//out.println("updatedWeightOutLayer("+i+"): "+UpdatedOutWeights[j] +", beforeUpdate: "+outLayer.getNeurons(i).getWeight(j) + ", errorWRTOUT: "+errorWRTOutput[i] + ", change: "+change +", hiddenOut0: " +hiddenOut[0] +", inut:"+inputs[0]);
				// save current weight change
				//out.println("OutLayerchange: "+change);
			}
		}
		
		// update hidden
		for (int i = 0; i < numHU; i++) {

			double[] UpdatedOutWeights = new double[inputs.length+1];
			// update hidden FCLayer weights
			for (int k = 0; k < inputs.length + 1; k++) {
				// get previous change in weight
				//double v = hiddenLayer.getNeurons(i).getV(k);

				double errorWRTweight;
				if(k == inputs.length){
					errorWRTweight = errorWRTHiddenOut[i]* hiddenLayer.getNeurons(i).bias;
				}else{
					errorWRTweight = errorWRTHiddenOut[i] * inputs[k];
				}

				//UpdatedOutWeights[k] = (hiddenLayer.getNeurons(i).getWeight(k)- errorWRTweight * learningRate) ;
				//double change = errorWRTweight;
				//TODO no need to uddate weights here, save average and add momentum and ADAM
				double change = errorWRTweight * learningRate +momentum * hiddenLayer.getNeurons(i).getV(k);
				hiddenDelta[i][k] += learningRate*change/batchsize;
				//UpdatedOutWeights[k] = hiddenLayer.getNeurons(i).getWeight(k) - learningRate*hiddenDelta[i][k];
				hiddenLayer.getNeurons(i).setV(change, k);
				//out.println("updatedWeightHiddenLayer("+i+"): "+UpdatedOutWeights[k] +", beforeUpdate: "+hiddenLayer.getNeurons(i).getWeight(k) + ", errorWRTOUT: "+errorWRTHiddenOut[i] + ", change: "+change +", hiddenOut0: " +hiddenOut[0] +", inut:"+inputs[0]);
			}
		}


		// update weight 2
		for(int i = 0; i < deltas_2.size(); i++){
			for(int j = 0; j < C1_layer.plates.length;j++){
				// ki kj controls window's index
				for(int ki = 0; ki < C2_layer.kernal_length; ki++){
					for(int kj = 0; kj < C2_layer.kernal_length; kj++){
						// ai aj controls matrix2's index
						double gradient = 0;
						for(int ai = 0; ai < C1_layer.plates[j].matrix2.length-C2_layer.kernal_length+1; ai++){
							for(int aj = 0; aj < C1_layer.plates[j].matrix2.length-C2_layer.kernal_length+1; aj++){;
							gradient += deltas_2.get(i)[ai][aj]*C1_layer.plates[j].matrix2[ai+ki][aj+kj];
							}
						}
						//TODO no need to uddate weights here, save average and add momentum and ADAM

						conv2Delta.get(i)[j][ki][kj] += gradient/batchsize;
						C2_layer.plates[i].kernal[j][ki][kj] -= learningRate *conv2Delta.get(i)[j][ki][kj];
					}

				}
			}
			// update bias delta
			for(int ai = 0; ai < c2_maxtrix1_len; ai++){
				for(int aj = 0; aj < c2_maxtrix1_len; aj++){
					//TODO no need to uddate weights here, save average and add momentum and ADAM
					double gradient = deltas_2.get(i)[ai][aj]*C2_layer.bias;
					conv2Bias[i] += gradient/batchsize;
					C2_layer.biasWeight[i] -= learningRate*conv2Bias[i];
				}
			}
		}


		//update weight 1
		for(int i = 0; i < deltas_1.size(); i++){
			for(int j = 0; j < v.size(); j++){
				for(int ki = 0; ki < C1_layer.kernal_length; ki++){
					for(int kj = 0; kj < C1_layer.kernal_length; kj++){
						double gradient = 0;
						for(int ai = 0; ai < v.get(j).length - C1_layer.kernal_length+1; ai++){
							for(int aj = 0; aj < v.get(j).length - C1_layer.kernal_length+1; aj++){
								gradient += deltas_1.get(i)[ai][aj]*v.get(j)[ai+ki][aj+kj];
							}
						}
						//TODO no need to uddate weights here, save average and add momentum and ADAM
						conv1Delta.get(i)[j][ki][kj] += gradient/batchsize;
						C1_layer.plates[i].kernal[j][ki][kj] -= gradient*conv1Delta.get(i)[j][ki][kj];
					}
				}
			}
			// update bias delta
			for(int ai = 0; ai < c1_maxtrix1_len; ai++){
				for(int aj = 0; aj < c1_maxtrix1_len; aj++){
					//TODO no need to uddate weights here, save average and add momentum and ADAM
					double gradient = deltas_1.get(i)[ai][aj]*C1_layer.bias;
					conv1Bias[i] += gradient/batchsize;
					C1_layer.biasWeight[i] -= learningRate*conv1Bias[i];
				}
			}

		}
	}

	// one of N encoding
	public static double[] getLabels(int label){
		// six categories
		double[] labels = new double[6];
		labels[label] = 1;
		return labels;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////

}
