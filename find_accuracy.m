function [acc] = find_accuracy(M, ground_truth)
    [row, col] = find(M);

    acc = size(intersect([col row], ground_truth, 'rows'), 1)/size(ground_truth, 1);

end
