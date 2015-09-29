/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaaccess;

import java.util.Collections;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author Teofebano, Andrey
 */
public class MyJ48 extends Classifier{
    // Attribute
    private MyJ48[] child;
    private Attribute attrSeparator;
    private double[] result;
    private double classValue;
    private Attribute classAttribute;
    private double splitValue;
    public double threshold;
    
    // Code
    public MyJ48(){
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        // Missing Class
        data = new Instances(data);
        data.deleteWithMissingClass();
        makeTree(data);
    }
    
    private double computeGR(Instances data, Attribute attr){
        return (computeIG(data, attr)/computeSI(data, attr));
    }
    
    private double computeSI(Instances data, Attribute attr){
       double SI = 0;
       Vector<Instances> instances = split(data,attr);
       for (int i=0;i<attr.numValues();i++){
           if (instances.elementAt(i).numInstances() > 0){
               SI += (instances.elementAt(i).numInstances() / data.numInstances()) * Utils.log2(instances.elementAt(i).numInstances() / data.numInstances());
           }
       }
       SI *= -1;
       return SI;
    }
    
    private double computeIG(Instances data, Attribute attr){
       double IG = computeEntropy(data);
       Vector<Instances> instances = split(data,attr);
       for (int i=0;i<attr.numValues();i++){
           if (instances.elementAt(i).numInstances() > 0){
               IG -= (instances.elementAt(i).numInstances() / data.numInstances()) * computeEntropy(instances.elementAt(i));
           }
       }
       return IG;
    }
    
    private double computeEntropy(Instances data){
        double entropy = 0;
        Vector<Double> classCounter = new Vector<Double>();
        classCounter.setSize(data.numClasses());
        for (int i=0;i<classCounter.size();i++){
            classCounter.setElementAt(Double.valueOf(0), i);
        }
        int numInstance = data.numInstances();
        for (int i=0;i<numInstance;i++){
            int cv = (int) data.instance(i).classValue();
            classCounter.setElementAt(classCounter.elementAt(cv)+1, cv);
        }
        for (int i=0;i<data.numClasses();i++){
            if (classCounter.elementAt(i)>0){
                entropy -= classCounter.elementAt(i) * Utils.log2(classCounter.elementAt(i));
            }
        }
        entropy /= (double) data.numInstances();
        return entropy;
    }
    
    private Vector<Instances> split(Instances data, Attribute attr){
        Vector<Instances> group = new Vector<Instances>(attr.numValues());
        if (attr.isNominal()){    
            for (int i = 0; i < attr.numValues(); i++) {
                group.add(new Instances(data, data.numInstances()));
            }
            for (int i=0;i<data.numInstances();i++){
                int av = (int) data.instance(i).value(attr);
                group.elementAt(av).add(data.instance(i));
            }
            for (int i=0;i<group.size();i++){
                group.elementAt(i).compactify();
            }
        }
        else{
            for (int i = 0; i < attr.numValues(); i++) {
                group.add(new Instances(data, data.numInstances()));
            }
            double threshold = countThreshold(data, attr);
            for (int i=0;i<data.numInstances();i++){
                int av = 0;
                if (data.instance(i).value(attr)<threshold){
                    av = 0;
                }
                else{
                    av = 1;
                }
                group.elementAt(av).add(data.instance(i));
            }
            for (int i=0;i<group.size();i++){
                group.elementAt(i).compactify();
            }
        }
        return group;
    }
    
    public Capabilities getCapabilities() {
        Capabilities capa = super.getCapabilities();
        capa.disableAll();
        
        capa.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        capa.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capa.enable(Capabilities.Capability.NOMINAL_CLASS);
        capa.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        capa.setMinimumNumberInstances(0);

        return capa;
    }
    
    private void makeTree(Instances trainingData) {
        if (trainingData.numInstances() == 0){
            attrSeparator = null;
            result = new double[trainingData.numClasses()];
            classValue = Instance.missingValue();
            threshold = Double.MIN_VALUE;
            return;
        }
        
        Vector<Double> listGR =  new Vector<Double>();
        listGR.setSize(trainingData.numAttributes());
        for (int i=0;i<listGR.size();i++){
            listGR.setElementAt(Double.valueOf(0), i);
        }
        for (int i=0;i<trainingData.numAttributes();i++){
            Attribute attr = trainingData.attribute(i);
            int attrIndex = attr.index();
            listGR.setElementAt(computeGR(trainingData, attr), attrIndex);
        }
        int index = listGR.indexOf(Collections.max(listGR));
        attrSeparator = trainingData.attribute(index);
        threshold = countThreshold(trainingData, attrSeparator);
        
        // Build Tree
        if (listGR.elementAt(index) == 0){ // leaf
            attrSeparator = null;
            result = new double[trainingData.numClasses()];
            for (int i=0;i<result.length;i++){
                result[(int)trainingData.instance(i).classValue()]++;
            }
            Utils.normalize(result);
            classValue = Utils.maxIndex(result);
            classAttribute = trainingData.classAttribute();
        }
        else{ // branch
            Vector<Instances> newData = split(trainingData,attrSeparator);
            child = new MyJ48[attrSeparator.numValues()];
            for (int i=0;i<child.length;i++){
                child[i] = new MyJ48();
                child[i].makeTree(newData.elementAt(i));
            }
        }
    }
    
    private void makeClassifier(Instances trainingData) throws Exception{
        getCapabilities().testWithFail(trainingData);
        trainingData.deleteWithMissingClass();
        makeTree(trainingData);
    }
    
    @Override
    public double classifyInstance(Instance testingData) throws NoSupportForMissingValuesException, Exception{
        if (testingData.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyJ48 can't handle such missing value");
        }
        if (attrSeparator == null){
            return classValue;
        }
        else{
            if (attrSeparator.isNominal()){
                return child[(int) testingData.value(attrSeparator)].classifyInstance(testingData);
            }
            else{
                int av = 0;
                if (testingData.value(attrSeparator)<threshold){
                    av = 0;
                }
                else{
                    av = 1;
                }
                return child[av].classifyInstance(testingData);
            }
        }
    }    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
          throw new NoSupportForMissingValuesException("MyJ48 can't handle such missing value");
        }
        if (attrSeparator == null) {
            return result;
        } else { 
            return child[(int) instance.value(attrSeparator)].distributionForInstance(instance);
        }
    }
    
    public double countThreshold(Instances trainingData, Attribute attr){
        Vector<Double> numericValue = new Vector<Double>();
        for (int j=0;j<trainingData.numInstances();j++){
            numericValue.add(trainingData.instance(j).value(attr));
        }
        sort(numericValue);
        boolean splitted = false;
        double threshold = Double.MIN_VALUE;
        for (int j=0;j<trainingData.numInstances()-1&&!splitted;j++){
            if (trainingData.instance(j).classValue()!= trainingData.instance(j+1).classValue()){
                splitted = true;
                threshold = (trainingData.instance(j).value(attr) + trainingData.instance(j+1).value(attr))/2;
            }
        }
        return threshold;
    }
    
    public void sort(Vector<Double> vector){
        Double temp;
        for (int i=0;i<vector.size()-1;i++){
            for (int j=1;j<vector.size()-i;j++){
                if (vector.elementAt(j-1)>vector.elementAt(j)){
                    temp = vector.elementAt(j-1);
                    vector.setElementAt(vector.elementAt(j), j-1);
                    vector.setElementAt(temp, j);
                }
            }
        }
    }
}
