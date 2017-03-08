import java.util.Vector;

public class Layer{
	Plate[] plates;
	Vector<double[][]> output_layer;
	int kernal_length;
	Vector<double[][]> kernals;
	double biasWeight[];
	double bias;
	int num_plate;
	int input_size;
	int pooling_length;
	double[] output1D;

	public Layer(int num_plate, int kernal_length, int pooling_length, int input_size){
		plates = new Plate[num_plate];
		output_layer = new Vector<double[][]>(num_plate);
		this.pooling_length = pooling_length;
		this.input_size = input_size;
		this.kernal_length = kernal_length;
		this.num_plate = num_plate;
		biasWeight = new double[num_plate];
		bias = -1;
		// init kernals
		this.kernals = new Vector<double[][]>(num_plate);
		
		for(int index = 0; index < num_plate; index++){
			//System.out.println("\nweights for "+ index + " : ");
			double[][] kernal = new double[kernal_length][kernal_length];
			for(int i = 0; i < kernal_length; i++){
				for(int j = 0; j < kernal_length; j++){		
					kernal[i][j] = getRandom(kernal_length*kernal_length+1,1);
				}
			}
			this.kernals.add(kernal);
		}

		
		// init bias
		for(int i = 0; i < biasWeight.length; i++){
			biasWeight[i] = getRandom(kernal_length*kernal_length+1,1);
		}

		// Edit  this is not image size
		int Clayer_length = input_size-kernal_length+1;

		for(int i = 0; i < num_plate; i++){
			plates[i] = new Plate(Clayer_length, kernals.get(i),kernal_length);
		}
		output1D = new double[num_plate*(Clayer_length/pooling_length)*(Clayer_length/pooling_length)];

	}
	// pass in one image //EDIT
	public Vector<double[][]> getOutput(Vector<double[][]> input){	


		// for all kernals
		for(int index = 0; index < kernals.size(); index++){
			double[][] kernal = kernals.get(index);

			// for all plates in the input layer
			for(int index_input = 0; index_input< input.size(); index_input++){
				//System.out.println("\nnetout "+ index + " : ");
				for(int i = 0; i < input_size-kernal_length+1; i++){
					for(int j = 0; j < input_size-kernal_length+1; j++){
						// multiply
						for(int ki = 0; ki < kernal_length; ki++){
							for(int kj = 0; kj < kernal_length; kj++){
								plates[index].matrix1[i][j] += input.get(index_input)[i+ki][j+kj]*kernal[ki][kj];
							}
						}
						//System.out.println(plates[index].matrix1[i][j]);
					}
				}
			}
			

			//EDIT 2  activation function
			for(int i = 0; i < input_size-kernal_length+1; i++){
				for(int j = 0; j < input_size-kernal_length+1; j++){
					double net_out = plates[index].matrix1[i][j]+(biasWeight[index]*bias);
					plates[index].inactivated[i][j] = net_out;
					plates[index].matrix1[i][j] = Math.max(0.0, net_out);
				}
			}
		}

		for(int i = 0; i < num_plate; i++){
			//System.out.println("\n"+i+"plate: ");
			plates[i].output(pooling_length);
			output_layer.add(plates[i].matrix2);
			//convert to 1d
			int len = plates[i].matrix2.length;
			for(int m =0;m<len;m++){
				for(int n =0;n<len;n++)
					output1D[n+m*len+ len*len] = plates[i].matrix1[m][n];
			}
		}
		return output_layer;
	}


	public double[] output1D(){
		return output1D;
	}




	private double getRandom(int fanin, int fanout){
		double range = Math.max(Double.MIN_VALUE, 4.0 / Math.sqrt(6.0 * (fanin + fanout)));
		return (2.0 * Lab3.random() - 1.0) * range;
	}


}