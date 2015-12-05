var assert = require('assert');

/** ======== Color-space manipulations ======== **/
global.RGBToYUV = function (rgb) {
	assert(rgb.r >= 0 && rgb.r <= 1);
	assert(rgb.g >= 0 && rgb.g <= 1);
	assert(rgb.b >= 0 && rgb.b <= 1);

	var yuv = {};
	yuv.y = Utils.clamp(0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b, 0, 1);
	yuv.u = Utils.clamp(-0.14713*rgb.r - 0.28886*rgb.g + 0.436*rgb.b, -U_MAX, U_MAX);
	yuv.v = Utils.clamp(0.615*rgb.r - 0.51499*rgb.g - 0.10001*rgb.b, -V_MAX, V_MAX);
	return yuv;
}

global.YUVToRGB = function (yuv) {
	assert(yuv.y >= 0 && yuv.y <= 1);
	assert(yuv.u >= -U_MAX && yuv.u <= U_MAX);
	assert(yuv.v >= -V_MAX && yuv.v <= V_MAX);

	var rgb = {};
	rgb.r = Utils.clamp(yuv.y + 1.13983*yuv.v, 0, 1);
	rgb.g = Utils.clamp(yuv.y - 0.39465*yuv.u - 0.58060*yuv.v, 0, 1);
	rgb.b = Utils.clamp(yuv.y + 2.03211*yuv.u, 0, 1);
	return rgb;
}


/** ======== Constants ======== **/
global.U_MAX = 0.436;
global.V_MAX = 0.615;

global.SQUARE_SIZE = 32;

/* Instead of training on every pixel in an image (and the square around it),
randomly sample this many pixles to train on. */
global.SAMPLES_PER_IMAGE = 200;

/** ======== Utilites ======== **/
global.Utils = {};
Utils.clamp = function (val, min, max) {
	return Math.max(min, Math.min(max, val));
}

// Returns a random integer between min (included) and max (excluded).
Utils.randInt = function (min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}