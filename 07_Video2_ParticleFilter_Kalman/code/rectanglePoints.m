function pts = rectanglePoints(cx, cy, W, H, n)
    x1 = cx - W/2;
    x2 = cx + W/2;
    y1 = cy - H/2;
    y2 = cy + H/2;

     top    = [linspace(x1,x2,n)' y1*ones(n,1)];
     bottom = [linspace(x1,x2,n)' y2*ones(n,1)];
    left   = [x1*ones(n,1) linspace(y1,y2,n)'];
    right  = [x2*ones(n,1) linspace(y1,y2,n)'];

     pts = [top; bottom; left; right];
    % pts = [left, right];
end