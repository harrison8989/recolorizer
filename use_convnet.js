var convnetjs = require("convnetjs");
var fs = require('fs');
var jimp = require('jimp');
var ndarray = require('ndarray');
var progress = require('progress');

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
	var YUV = ndarray(new Float64Array(numPixels), [image.bitmap.width, image.bitmap.height, 3]);

	console.log('> Convert to YUV...');
	for (var x = 0; x < image.bitmap.width; ++x) {
		for (var y = 0; y < image.bitmap.height; ++y) {
			// Get the color as RGB floats from 0-1
			var color = jimp.intToRGBA(image.getPixelColor(x, y));
			color.r = color.r / 255.0;
			color.g = color.g / 255.0;
			color.b = color.b / 255.0;
			delete color.a;

			// Convert to RGB and set in ndarray.
			var yuv = RGBToYUV(color);
			YUV.set(x, y, 0, yuv.y);
			YUV.set(x, y, 1, yuv.u);
			YUV.set(x, y, 2, yuv.v);
		}
	}

	var total = (image.bitmap.width - SQUARE_SIZE)*(image.bitmap.height - SQUARE_SIZE);
	var bar = new progress('Colorizing... [:bar] :percent :etas', {total: total});

	for(var x = SQUARE_SIZE/2 ; x < image.bitmap.width - SQUARE_SIZE/2 ; ++x) {
		for (var y = SQUARE_SIZE/ 2; y < image.bitmap.height - SQUARE_SIZE/2; ++y) {
			var square = YUV.lo(x - SQUARE_SIZE/2, y - SQUARE_SIZE/2, 0)
				.hi(SQUARE_SIZE, SQUARE_SIZE, 3);

			var input = new convnetjs.Vol(SQUARE_SIZE, SQUARE_SIZE, 1);
			for (var i = 0; i < SQUARE_SIZE; ++i) {
				for (var j = 0; j < SQUARE_SIZE; ++j) {
					input.set(i, j, 0, square.get(i, j, 0));
				}
			}

			var predicted = net.forward(input);
			var u = Utils.clamp(predicted.get(0,0,0), -U_MAX, U_MAX);
			var v = Utils.clamp(predicted.get(0,0,0), -V_MAX, V_MAX);

			var center = SQUARE_SIZE/2;
			var rgb = YUVToRGB({
				y: square.get(center, center, 0),
				u: u,
				v: v
			});

			colorized.setPixelColor(jimp.rgbaToInt(Math.floor(rgb.r*255), Math.floor(rgb.g*255), Math.floor(rgb.b*255), 255), x, y);
			bar.tick();
		}
	}

	colorized.write( "out.jpg", function() {
		console.log("Wrote image...");
	});
});