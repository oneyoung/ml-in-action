# desicion tree
import math


def entroy(dataset):
    '''
    get Shannon Entroy of dataset
    higher Entroy means more chaos

    params:
        dataset: NxM array data, the last element of each row is the result of
        category

    return Shannon Entroy
    '''
    total = len(dataset)
    counts = {}
    for vec in dataset:
        label = vec[-1]
        counts[label] = counts.get(label, 0) + 1
    ent = 0
    for label, count in counts.items():
        prob = float(count)/total
        ent -= prob*math.log(prob, 2)
    return ent


def split_dataset(dataset, index, value):
    '''
    split dataset
    filter out data vec with given index and value

    params:
        dataset: dataset to be processed
        index: index of vec to check
        value: target value to match

    return
    '''
    ret_data = []
    for vec in dataset:
        if vec[index] == value:
            # pop up value at index
            # then we can reduce complexity by 1 axes
            reduced_vec = vec[:index]
            reduced_vec.extend(vec[index+1:])
            ret_data.append(reduced_vec)
    return ret_data


def best_feature_split(dataset):
    '''
    choose best feature to split

    params:
        dataset: input dataset, the last member of vec is category result

    return: the index of best feature for split in vec
    '''
    feature_num = len(dataset[0]) - 1  # exclude result column
    base_entroy = entroy(dataset)
    best_index = -1
    best_info_gain = -1e100  # make base gain as infinate min
    for i in range(feature_num):
        # all possible values at index i
        values = set([vec[i] for vec in dataset])
        ientroy = 0  # total entroy of this split
        for v in values:
            sub_set = split_dataset(dataset, i, v)
            # the length of sub_set is how many time value=v appears
            # so probility is len(sub_set)/ (total samples)
            prob = len(sub_set)/float(len(dataset))
            ientroy += entroy(sub_set)
        info_gain = base_entroy - ientroy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_index = i
    return best_index


def get_majority(labels):
    '''
    get the majority lable (which highest count) of labels list
    '''
    count_map = {}
    for label in labels:
        count_map = count_map.get(label, 0) + 1
    major = None
    major_count = 0
    for label, count in count_map.items():
        if count > major_count:
            major_count = count
            major = label
    return major


def create_tree(dataset, input_labels):
    '''
    create decision tree

    params:
        dataset: NxM input array, last member of each vec is category result
        labels: literal label of eash feature, length: (M-1)

    return: desicion tree
    '''
    labels = input_labels[:]
    categories = [vec[-1] for vec in dataset]
    # terminate condition:
    if categories.count(categories[0]) == len(categories):
        # already category to the same result
        return categories[0]
    if len(dataset[0]) == 1:
        # no feature left, only label exists in vec
        # can't split any more, choose the majority of category
        return get_majority(categories)
    # tree create
    best_index = best_feature_split(dataset)  # get best split feature
    # literal label for that feature, and trim to sub-labels for iter
    best_label = labels.pop(best_index)
    tree = {best_label: {}}
    unique_feature_values = set([vec[best_index] for vec in dataset])
    for value in unique_feature_values:
        sub_labels = labels[:]  # copy instead of reference
        tree[best_label][value] = create_tree(
            split_dataset(dataset, best_index, value),
            sub_labels)  # iter to sub tree
    return tree


def classify(tree, feature_labels, input_vec):
    '''
    classify using decision tree

    params:
        tree: decision tree
        feature_labels: feature labels list
        input_vec: input vec
    '''
    root = tree.keys()[0]
    sub_dict = tree[root]
    index = feature_labels.index(root)  # feature index
    for key, sub_node in sub_dict.iteritems():
        if input_vec[index] == key:  # tree matched
            if isinstance(sub_node, dict):  # is sub tree, iter
                return classify(sub_node, feature_labels, input_vec)
            else:  # leaf, return value
                return sub_node
