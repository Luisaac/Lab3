import java.util.Vector;

public class Layer{
	Plate[] plates;
	Vector<double[][]> output_layer;
	int kernal_length;
	Vector<double[][][]> kernals;
	double biasWeight[];
	double bias;
	int num_plate;
	int input_size;
	int pooling_length;
	double[] output1D;

	public Layer(int num_input_plate, int num_plate, int kernal_length, int pooling_length, int input_size){
		plates = new Plate[num_plate];
		output_layer = new Vector<double[][]>(num_plate);
		this.pooling_length = pooling_length;
		this.input_size = input_size;
		this.kernal_length = kernal_length;
		this.num_plate = num_plate;
		biasWeight = new double[num_plate];
		bias = -1;
		// init kernals
		this.kernals = new Vector<double[][][]>(num_plate);

		for(int index = 0; index < num_plate; index++){		
			double[][][] kernal = new double[num_input_plate][kernal_length][kernal_length];
			for(int a = 0; a < num_input_plate; a++){
				for(int i = 0; i < kernal_length; i++){
					for(int j = 0; j < kernal_length; j++){		
						//TODO
						kernal[a][i][j] = getRandom(kernal_length*kernal_length*num_input_plate+1,0);
						//kernal[a][i][j] = 1.0;
					}
				}
			}
			this.kernals.add(kernal);
		}


		// init bias
		for(int i = 0; i < biasWeight.length; i++){
			//TODO
			biasWeight[i] = getRandom(kernal_length*kernal_length*num_input_plate+1,0);
			//biasWeight[i] = 0;
		}

		// Edit  this is not image size
		int Clayer_length = input_size-kernal_length+1;

		for(int i = 0; i < num_plate; i++){
			plates[i] = new Plate(Clayer_length, kernals.get(i),kernal_length);
		}
		//TODO
		output1D = new double[num_plate*(Clayer_length/pooling_length)*(Clayer_length/pooling_length)];
		//output1D = new double[num_plate*(Clayer_length)*(Clayer_length)];
	}

	// pass in one image //EDIT
	public Vector<double[][]> feedForward(Vector<double[][]> input){
		for(int index = 0; index<num_plate; index++){
			for(int i = 0; i < plates[index].matrix1.length; i++){
				for(int j = 0; j < plates[index].matrix1.length; j++){
					plates[index].matrix1[i][j] = 0;
				}
			}
		}

		//System.out.println("input size: "+input.size() + ", length: "+input.get(0).length +", kernal size: "+kernal_length);
		// for all kernals
		for(int index = 0; index < kernals.size(); index++){
			double[][][] kernal = kernals.get(index);

			// for all plates in the input layer
			double c =0;
			for(int i = 0; i < input_size-kernal_length+1; i++){
				for(int j = 0; j < input_size-kernal_length+1; j++){
					for(int index_input = 0; index_input< input.size(); index_input++){
						// multiply
						for(int ki = 0; ki < kernal_length; ki++){
							for(int kj = 0; kj < kernal_length; kj++){
								plates[index].matrix1[i][j] += input.get(index_input)[i+ki][j+kj]*kernal[index_input][ki][kj];
							}
						}
						
					}
//					if(index == 0){
//						c++;
//						System.out.println("matrix1: "+plates[index].matrix1[i][j] + ", c: "+ c + ", plate:"+index);
//					}
				}

			}
		}


		//EDIT 2  activation function
		for(int index = 0; index < plates.length;index++){
			//		System.out.println("\nnetout "+ index + " : ");
			for(int i = 0; i < input_size-kernal_length+1; i++){
				for(int j = 0; j < input_size-kernal_length+1; j++){
					double net_out = plates[index].matrix1[i][j] + biasWeight[index]*bias;

//										if(index == 0 && kernals.get(0).length == 20){
//											System.out.println("net_out:"+net_out);
//					
//										}
					plates[index].inactivated[i][j] = net_out;
					plates[index].matrix1[i][j] = net_out>0?net_out:0.01*net_out;	
				}
			}
		}
		
		output_layer.clear();
		for(int i = 0; i < num_plate; i++){
			//System.out.println("\n"+i+"plate: ");
			plates[i].output(pooling_length);
			output_layer.add(plates[i].matrix2);
		}
		return output_layer;
	}

	public Vector<double[][]> getOutput(){

		return output_layer;
	}

	public void clearMax(){
		for(int i =0;i<plates.length;i++){
			int len = plates[i].useAsMax.length;
			for(int m = 0; m<len;m++){
				for(int n =0;n<len;n++){
					plates[i].useAsMax[m][n] = false;
				}
			}
		}
	}

	public double[] output1D(){
		for(int i = 0; i < num_plate; i++){
			//convert to 1d
			int len = plates[i].matrix2.length;
			for(int m =0;m<len;m++){
				for(int n =0;n<len;n++){
					double a = plates[i].matrix2[m][n];
					output1D[n+m*len+len*len*i] = plates[i].matrix2[m][n];
				}
			}
		}
		
		return output1D;
	}

	private double getRandom(int fanin, int fanout){
		double range = Math.max(Double.MIN_VALUE, 1.0 / Math.sqrt(fanin + fanout));
		return (2.0 * Lab3.random() - 1.0) * range;
		//return -0.03+0.06*Lab3.random();
	}


}