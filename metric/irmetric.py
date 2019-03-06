class Metric:
    def test(self, prediction, ground):
        return self.false_negatives(prediction, ground)
    
    def precision(self, prediction, ground):
        true_positive = self.true_positives(prediction, ground);
        if true_positive == 0:
            return 0
        false_positive = self.false_positives(prediction, ground);
        return true_positive / (true_positive + false_positive)
    
    def recall(self, prediction, ground):
        true_positive = self.true_positives(prediction, ground);
        
        if true_positive == 0:
            return 0
        
        false_negatives = self.false_negatives(prediction, ground);
        return true_positive / (true_positive + false_negatives)
    
    def f1(self, prediction, ground):
        precision = self.precision(prediction, ground)
        recall = self.recall(prediction, ground)
        if precision == 0 or recall ==0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def true_positives(self, prediction, ground):
        assert(self.__is_validated(prediction, ground) == True)
        true_positive = 0
        for i in range(len(prediction)):
            if prediction[i] == 1 and ground[i] == 1:
                true_positive += 1
        return true_positive
    
    def false_positives(self, prediction, ground):
        assert(self.__is_validated(prediction, ground) == True)
        false_positive = 0
        for i in range(len(prediction)):
            if prediction[i] == 1 and ground[i] == 0:
                false_positive += 1
        return false_positive    
    
    def false_negatives(self, prediction, ground):
        assert(self.__is_validated(prediction, ground) == True)
        false_negatives = 0
        for i in range(len(prediction)):
            if prediction[i] == 0 and ground[i] == 1:
                false_negatives += 1
        return false_negatives     

    def __is_validated(self, prediction, ground):
        assert(len(prediction) == len(ground))
        assert(len(prediction) != 0)
        return True
        
