#import "TextDetectorManager.h"
#if __has_include(<MLKitTextRecognition/MLKitTextRecognition.h>)
@import MLKitVision;

@interface TextDetectorManager ()
@property(nonatomic, strong) MLKTextRecognizer *textRecognizer;
@property(nonatomic, assign) float scaleX;
@property(nonatomic, assign) float scaleY;
@end

@implementation TextDetectorManager

- (instancetype)init
{
  if (self = [super init]) {
    self.textRecognizer = [MLKTextRecognizer textRecognizer];
  }
  return self;
}

- (BOOL)isRealDetector
{
  return true;
}

- (void)findTextBlocksInFrame:(UIImage *)uiImage scaleX:(float)scaleX scaleY:(float) scaleY completed: (void (^)(NSArray * result)) completed
{
    self.scaleX = scaleX;
    self.scaleY = scaleY;
    MLKVisionImage *visionImage = [[MLKVisionImage alloc] initWithImage:uiImage];
    NSMutableArray *textBlocks = [[NSMutableArray alloc] init];
    [_textRecognizer processImage:visionImage
                       completion:^(MLKText *_Nullable result,
                                    NSError *_Nullable error) {
                           if (error != nil || result == nil) {
                               completed(textBlocks);
                           } else {
                               completed([self processBlocks:result.blocks]);
                           }
                       }];
}

- (NSArray *)processBlocks:(NSArray *)features
{
  NSMutableArray *textBlocks = [[NSMutableArray alloc] init];
  for (MLKTextBlock *textBlock in features) {
      NSDictionary *textBlockDict = 
      @{@"type": @"block", @"value" : textBlock.text, @"bounds" : [self processBounds:textBlock.frame], @"components" : [self processLine:textBlock.lines]};
      [textBlocks addObject:textBlockDict];
  }
  return textBlocks;
}

- (NSArray *)processLine:(NSArray *)lines
{
  NSMutableArray *lineBlocks = [[NSMutableArray alloc] init];
  for (MLKTextLine *textLine in lines) {
        NSDictionary *textLineDict = 
        @{@"type": @"line", @"value" : textLine.text, @"bounds" : [self processBounds:textLine.frame], @"components" : [self processElement:textLine.elements]};
        [lineBlocks addObject:textLineDict];
  }
  return lineBlocks;
}

- (NSArray *)processElement:(NSArray *)elements
{
  NSMutableArray *elementBlocks = [[NSMutableArray alloc] init];
  for (MLKTextElement *textElement in elements) {
        NSDictionary *textElementDict = 
        @{@"type": @"element", @"value" : textElement.text, @"bounds" : [self processBounds:textElement.frame]};
        [elementBlocks addObject:textElementDict];
  }
  return elementBlocks;
}

- (NSDictionary *)processBounds:(CGRect)bounds
{
  float width = bounds.size.width * _scaleX;
  float height = bounds.size.height * _scaleY;
  float originX = bounds.origin.x * _scaleX;
  float originY = bounds.origin.y * _scaleY;
  NSDictionary *boundsDict =
  @{
    @"size" : 
              @{
                @"width" : @(width), 
                @"height" : @(height)
                }, 
    @"origin" : 
              @{
                @"x" : @(originX),
                @"y" : @(originY)
                }
    };
  return boundsDict;
}

@end
#else
#import <Vision/Vision.h>

@interface TextDetectorManager ()
@property(nonatomic, assign) float scaleX;
@property(nonatomic, assign) float scaleY;
@end

@implementation TextDetectorManager

- (instancetype)init
{
    if (self = [super init]) {
    }
    return self;
}

-(BOOL)isRealDetector
{
    if (@available(iOS 13.0, *)) {
        return true;
    } else {
        return false;
    }
}

- (void)findTextBlocksInFrame:(UIImage *)uiImage scaleX:(float)scaleX scaleY:(float) scaleY completed: (void (^)(NSArray * result)) completed
{
    if (@available(iOS 13.0, *)) {
        self.scaleX = scaleX;
        self.scaleY = scaleY;
        
        CGSize size = uiImage.size;
        
        VNRecognizeTextRequest *textRequest = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
            if (error != nil) {
                completed(@[]);
                return;
            }
            
            NSArray<VNRecognizedTextObservation *> * textObservations = request.results;
            NSMutableArray *results = [NSMutableArray new];
            
            for (VNRecognizedTextObservation *textObservation in textObservations) {
                VNRecognizedText *text = [[textObservation topCandidates:1] firstObject];
                if (text == nil || text.confidence < 0.5) {
                    continue;
                }
                
                CGFloat minX = MIN(MIN(MIN(textObservation.topLeft.x, textObservation.bottomLeft.x), textObservation.topRight.x), textObservation.bottomRight.x) * size.width;
                CGFloat maxX = MAX(MAX(MAX(textObservation.topLeft.x, textObservation.bottomLeft.x), textObservation.topRight.x), textObservation.bottomRight.x) * size.width;
                
                CGFloat minY = MIN(MIN(MIN(textObservation.bottomLeft.y, textObservation.bottomRight.y), textObservation.topLeft.y), textObservation.topRight.y) * size.height;
                CGFloat maxY = MAX(MAX(MAX(textObservation.bottomLeft.y, textObservation.bottomRight.y), textObservation.topLeft.y), textObservation.topRight.y) * size.height;
                
                
                [results addObject:@{
                    @"type": @"block",
                    @"value": text.string,
                    @"bounds": [self processBounds:CGRectMake(minX, size.height - maxY, maxX - minX, maxY - minY)]
                }];
            }
            
            completed(results);
            
        }];
        
        textRequest.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        
        // Language correction won't help recognizing phone numbers. It also
        // makes recognition slower.
        textRequest.usesLanguageCorrection = NO;
        
        VNImageRequestHandler *h = [[VNImageRequestHandler alloc] initWithCGImage:uiImage.CGImage options:@{}];
        
        
        [h performRequests:@[textRequest] error:nil];
    } else {
        NSLog(@"TextDetector not installed, stub used!");
        NSArray *features = @[@"Error, Text Detector not installed"];
        completed(features);
        return;
    }
}

-(NSDictionary *)processBounds:(CGRect)bounds
{
    float width = bounds.size.width * _scaleX;
    float height = bounds.size.height * _scaleY;
    float originX = bounds.origin.x * _scaleX;
    float originY = bounds.origin.y * _scaleY;
    NSDictionary *boundsDict =
    @{
        @"size" :
            @{
                @"width" : @(width),
                @"height" : @(height)
            },
        @"origin" :
            @{
                @"x" : @(originX),
                @"y" : @(originY)
            }
    };
    return boundsDict;
}

@end

#endif
