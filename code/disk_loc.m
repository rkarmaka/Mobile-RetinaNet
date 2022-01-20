function [out_img, out_mask, stats]=disk_loc(img,mask)
    se=strel('disk',8,4);
    im_cl=imclose(img(:,:,2),se);
    im_ad=imadjust(im_cl);
    out=(im_ad>250);
    out=imopen(out,se);
    
    stats=regionprops(out,'centroid','area');
    
    xmin = stats(1).Centroid(1)-128;
    xmin
    if xmin<1
        xmin=1
    end
    ymin = stats(1).Centroid(2)-128;
    out_img = imcrop(img,[xmin ymin 256 256]);
    out_mask = imcrop(mask,[xmin ymin 256 256]);
%     out_img=out;
%     out_mask=out;
end