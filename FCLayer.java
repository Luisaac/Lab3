import java.util.ArrayList;
import java.util.List;

public class FCLayer {
	public List<Neuron> neurons;
	private double[] outputs;
	private int size;

	public FCLayer(int numNeurons,int numLinks){
		neurons = new ArrayList<Neuron>();
		size = numNeurons;
		// one biased term
		outputs = new double[numNeurons];
		for(int i = 0;i<numNeurons;i++){
			Neuron tmp = new Neuron(numLinks);
			neurons.add(tmp);
		}
	}

//	public FCLayer(FCLayer oldFCLayer){
//		neurons = new ArrayList<Neuron>();
//		this.isSigmoid = oldFCLayer.isSigmoid;
//	}

	public double[] feedForward(double[] inputs){
		
		for(int i = 0;i<size;i++){
			outputs[i] = neurons.get(i).activatedOutput(inputs);
			
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

