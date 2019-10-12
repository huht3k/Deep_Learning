x = rand(3, 10);
[n, m] = size(x);

cov1 = cov(x');

avg = mean(x, 2);
x1 = x - repmat(avg,1, m);

cov2 = x1 * x1' / ((size(x1, 2)) - 1);