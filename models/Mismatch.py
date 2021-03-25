import numpy as np

DNA_TO_DIGIT = {'A': 0,
                'C': 1,
                'G': 2,
                'T': 3}


class MismatchTree:

    def __init__(self, label=None, parent=None):
        self.label = label
        self.cum_label = ""
        self.parent = parent
        self.depth = 0
        self.children = {}
        self.set_ptrs = {}
        if parent is not None:
            parent.add_node(self)
    
    def add_node(self, child):
        child.cum_label = self.cum_label + str(child.label)
        child.parent = self
        child.depth = self.depth + 1
        self.children[child.label] = child
        child.set_ptrs = {ind: np.array(sub_ptrs) for ind, sub_ptrs in self.set_ptrs.items()}
    
    def delete_node(self, child):
        # get child label
        label = child.label if isinstance(child, MismatchTree) else child

        # check that child really exists
        assert label in self.children, "No child with label %s exists." % label

        # delete the child
        del self.children[label]

    def init_ptrs_at_root(self, train_data, k):
        for i in range(len(train_data)):
            self.set_ptrs[i] = np.array([(0, substr_ind) for substr_ind in range(len(train_data[i]) - k + 1)])

    def process(self, train_data, k, m):
        # if root
        if self.parent is None:
            self.init_ptrs_at_root(train_data, k)
        else:
            for ind, sub_ptrs in self.set_ptrs.items():
                # update
                sub_ptrs[:, 0] += (train_data[ind][sub_ptrs[:, 1] + self.depth - 1] != self.label)

                # delete more than m mismatches
                self.set_ptrs[ind] = np.delete(sub_ptrs, np.nonzero(sub_ptrs[..., 0] > m), axis=0)
        
        # delete empty
        self.set_ptrs = {index: substring_pointers for index, substring_pointers in self.set_ptrs.items() if len(substring_pointers)}
        
        return len(self.set_ptrs) != 0

    def traverse(self, train_data, k, m, l, kernel=None):
        """
        Recursive traversal of the Mismatch Tree
        """
        if kernel is None:
            num_samples = train_data.shape[0]
            kernel = np.zeros((num_samples, num_samples))

        keep_going = self.process(train_data, k, m)

        if keep_going:
            # reach a leaf -> k-mer
            if k == 0:
                print(self.cum_label)
                for i in self.set_ptrs.keys():
                    for j in self.set_ptrs.keys():
                        # update kernel_ij
                        kernel[i, j] += len(self.set_ptrs[i]) * len(self.set_ptrs[j])
            else:
                for j in range(l):
                    child = MismatchTree(label=j, parent=self)
                    kernel = child.traverse(train_data, k - 1, m, l, kernel=kernel)

                    if len(child.set_ptrs) == 0:
                        self.delete_node(child)

        return kernel


class MismatchKernelForBioSeq:

    def __init__(self, train_data, k, m, l):
        self.train_data = train_data
        self.k, self.m, self.l = k, m, l
        self.kernel = None
        self.tree = MismatchTree()
        self.preprocess()
        print(self.train_data)

    def preprocess(self):
        length_seq = len(self.train_data[0])
        new_train_data = np.zeros((self.train_data.shape[0], length_seq))
        for i in range(self.train_data.shape[0]):
            for j in range(length_seq):
                new_train_data[i, j] = DNA_TO_DIGIT[self.train_data[i][j]]
        
        self.train_data = new_train_data

    def compute_kernel(self):
        kernel = self.tree.traverse(self.train_data, self.k, self.m, self.l)
        self.kernel = kernel

        return kernel
    
    def normalize_kernel(self, kernel):
        nkernel = np.copy(kernel)

        assert nkernel.ndim == 2
        assert nkernel.shape[0] == nkernel.shape[1]

        for i in range(nkernel.shape[0]):
            for j in range(i + 1, nkernel.shape[0]):
                q = np.sqrt(nkernel[i, i] * nkernel[j, j])
                if q > 0:
                    nkernel[i, j] /= q
                    nkernel[j, i] = nkernel[i, j]  # symmetry

        # Set diagonal elements as 1
        np.fill_diagonal(nkernel, 1.)

        return nkernel