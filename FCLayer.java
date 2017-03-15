import java.util.ArrayList;
import java.util.List;

public class FCLayer {
	public List<Neuron> neurons;
	private double[] outputs;
	private int size;
	private int actfunc;
	int predicted;
	public FCLayer(int numNeurons,int numLinks, int actfunc){
		neurons = new ArrayList<Neuron>();
		size = numNeurons;
		// one biased term
		outputs = new double[numNeurons];
		for(int i = 0;i<numNeurons;i++){
			Neuron tmp = new Neuron(numLinks, actfunc);
			neurons.add(tmp);
		}
		this.actfunc = actfunc;
		predicted = -1;
	}


	public double[] feedForward(double[] inputs){
		
		if(actfunc == 1){
			double max = Double.NEGATIVE_INFINITY;
			double sum = 0;
			for(int i = 0;i<size;i++){
				double tmp = neurons.get(i).activatedOutput(inputs);
				if(tmp>max){
					max = tmp;
					predicted = i;
				}
				outputs[i] = tmp;
				//System.out.println("before squashing: "+tmp);
				//outputs[i] = neurons.get(i).activatedOutput(inputs);
			}
			for(int i=0;i<size;i++){			
				outputs[i] = Math.exp(outputs[i]-max);
				sum += outputs[i];
			}
			for(int i=0;i<size;i++){			
				outputs[i] /= sum;
			}
		}else{
			for(int i=0;i<size;i++){
				
				outputs[i] = neurons.get(i).activatedOutput(inputs);
				//System.out.println("output:"+outputs[i]);
			}
		}
		return outputs;
	}

	public double[] getOutputs(){
		return outputs;
	}

	// return the size of the FCLayer
	public int size(){
		return size;
	}

	
	public void clearNeurons() {
		neurons.clear();
	}

	public Neuron getNeurons(int i){
		return neurons.get(i);
	}


	public List<Neuron> getListOfNeurons() {
		return neurons;

	}

	public void addNeuron ( Neuron e) {
		neurons.add(e);
	}




}

