function stretched = myContrastStretch(img)
    img = double(img);
    minVal = min(img(:));
    maxVal = max(img(:));

    if maxVal == minVal
        stretched = img;
    else
        stretched = (img - minVal) * 255 / (maxVal - minVal);
    end

    stretched = uint8(stretched);
end
