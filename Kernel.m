function k = Kernel(x , y , type)

switch type
    case 'rbf'
        p = 1;
        k = exp(-norm(x-y)^2/(2*p^2));
        
    
end

end

