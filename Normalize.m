function X = Normalize(X)
X = (X - min(X)) ./ (max(X) - min(X));

end

