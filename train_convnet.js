var convnetjs = require("convnetjs");
var fs = require('fs');
var jimp = require('jimp');
var walk = require('fs-walk');
var path = require('path');
var ndarray = require('ndarray');
require('./globals.js');

var layer_defs = [];
layer_defs.push({type:'input', out_sx: SQUARE_SIZE, out_sy: SQUARE_SIZE, out_depth: 1});
layer_defs.push({type:'conv', sx: 5, filters: 16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx: 2, stride: 2});
layer_defs.push({type:'conv', sx: 5, filters: 20, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx: 2, stride: 2});
layer_defs.push({type:'conv', sx: 5, filters: 20, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx: 2, stride: 2});
layer_defs.push({type:'regression', num_neurons: 2});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// example that uses adadelta. Reasonable for beginners.
var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: SAMPLES_PER_IMAGE*3});

// Train.
var promise = Promise.resolve();

walk.walkSync('data/flickr/', function (basedir, filename, stat) {
	if (filename.endsWith('.jpg')) {
		promise = promise.then(function() {
			return new Promise(function(resolve, reject) {
				jimp.read(path.join(basedir, filename), function (err, image) {
					if (err) {
						console.error(err);
						process.exit();
					}
					console.log("Training on " + filename + " (" + image.bitmap.width
						+ "x" + image.bitmap.height + ") ...");
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

					console.log("> Randomly training on sampled subsquares...");
					for (var n = 0; n < SAMPLES_PER_IMAGE; ++n) {
						var x = Utils.randInt(SQUARE_SIZE/2, image.bitmap.width - SQUARE_SIZE/2);
						var y = Utils.randInt(SQUARE_SIZE/2, image.bitmap.height - SQUARE_SIZE/2);

						var square = YUV.lo(x - SQUARE_SIZE/2, y - SQUARE_SIZE/2, 0)
							.hi(SQUARE_SIZE, SQUARE_SIZE, 3);

						var input = new convnetjs.Vol(SQUARE_SIZE, SQUARE_SIZE, 1);
						for (var i = 0; i < SQUARE_SIZE; ++i) {
							for (var j = 0; j < SQUARE_SIZE; ++j) {
								input.set(i, j, 0, square.get(i, j, 0));
							}
						}
						var output = [square.get(SQUARE_SIZE/2, SQUARE_SIZE/2, 1), square.get(SQUARE_SIZE/2, SQUARE_SIZE/2, 2)];
						var stats = trainer.train(input, output);
					}
					
					resolve();
				});
			});
		});
	}
});


promise.then(function() {
	console.log("Finished. Saving neural network to 'convnet.json'.");

	// Save model.
	var json = net.toJSON();
	fs.writeFileSync('convnet.json', JSON.stringify(json), 'utf-8');
}).catch(function(e) {
	console.error(e);
});
