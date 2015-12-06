var assert = require('assert');

/** ======== Color-space manipulations ======== **/
global.RGBToYUV = function (rgb) {
	var yuv = {};
	yuv.y = Utils.clamp(0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b, 0, 1);
	yuv.u = Utils.clamp(-0.147*rgb.r - 0.289*rgb.g + 0.436*rgb.b, -U_MAX, U_MAX);
	yuv.v = Utils.clamp(0.615*rgb.r - 0.515*rgb.g - 0.100*rgb.b, -V_MAX, V_MAX);
	return yuv;
}

global.YUVToRGB = function (yuv) {
	var rgb = {};
	rgb.r = Utils.clamp(yuv.y + 1.140*yuv.v, 0, 1);
	rgb.g = Utils.clamp(yuv.y - 0.395*yuv.u - 0.581*yuv.v, 0, 1);
	rgb.b = Utils.clamp(yuv.y + 2.032*yuv.u, 0, 1);
	return rgb;
}


/** ======== Constants ======== **/
global.U_MAX = 0.436;
global.V_MAX = 0.615;

global.SQUARE_SIZE = 10;

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

/** ========= Polyfills ======== **/

if (!String.prototype.endsWith) {
  String.prototype.endsWith = function(searchString, position) {
      var subjectString = this.toString();
      if (typeof position !== 'number' || !isFinite(position) || Math.floor(position) !== position || position > subjectString.length) {
        position = subjectString.length;
      }
      position -= searchString.length;
      var lastIndex = subjectString.indexOf(searchString, position);
      return lastIndex !== -1 && lastIndex === position;
  };
}
