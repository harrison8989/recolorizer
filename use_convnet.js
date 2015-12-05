var convnetjs = require("convnetjs");
var fs = require('fs');
var jimp = require('jimp');
var ndarray = require('ndarray');
require('./globals.js');


console.log("Loading network from file...");
var network_json = JSON.parse(fs.readFileSync('convnet.json'));
var net = new convnetjs.Net();
net.fromJSON(network_json);

console.log("Loading image...");
var filename = process.argv[2];
jimp.read(filename, function (err, image) {
	if (err) {
		console.error(err);
		return;
	}

	var colorized = image.clone();
	var numPixels = image.bitmap.width * image.bitmap.height * 3;
	var YUV = ndarray(new Float32Array(numPixels), [image.bitmap.width, image.bitmap.height, 3]);

	console.log('> Convert to YUV...');
	for (var x = 0; x < image.bitmap.width; ++x) {
		for (var y = 0; y < image.bitmap.height; ++y) {
			var color = jimp.intToRGBA(image.getPixelColor(x, y));
			color.r = color.r / 255.0;
			color.g = color.g / 255.0;
			color.b = color.b / 255.0;
			delete color.a;
			var yuv = RGBToYUV(color);

			YUV.set(x, y, 0, yuv.y);
			YUV.set(x, y, 1, yuv.u);
			YUV.set(x, y, 2, yuv.v);
		}
	}

	console.log('> Colorizing...');
	for(var x = SQUARE_SIZE; x < image.bitmap.width - SQUARE_SIZE; ++x) {
		for (var y = SQUARE_SIZE; y < image.bitmap.height - SQUARE_SIZE; ++y) {
			var square = YUV.lo(x - SQUARE_SIZE/2, y - SQUARE_SIZE/2, 0)
						.hi(SQUARE_SIZE, SQUARE_SIZE, 3);

			/*var input = new convnetjs.Vol(SQUARE_SIZE, SQUARE_SIZE, 1);
			for (var i = 0; i < SQUARE_SIZE; ++i) {
				for (var j = 0; j < SQUARE_SIZE; ++j) {
					input.set(i, j, 0, square.get(i, j, 0));
				}
			}
			var predicted = net.forward(input);
			var u = Utils.clamp(predicted.get(0,0,0), -U_MAX, U_MAX);
			var v = Utils.clamp(predicted.get(0,0,0), -V_MAX, V_MAX);*/

			var center = SQUARE_SIZE/2;
			var rgb = YUVToRGB({
				y: square.get(center, center, 0),
				u: square.get(center, center, 1),
				v: square.get(center, center, 2)
			});

			colorized.setPixelColor(jimp.rgbaToInt(rgb.r*255, rgb.g*255, rgb.b*255, 255), x, y);

		}
	}

	colorized.write( "out.jpg", function() {
		console.log("Wrote image...");
	});
});