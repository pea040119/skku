INSERT INTO Users(user_name, pwd, is_admin) VALUES ('pea', 'pea_db', true);
INSERT INTO Users(user_name, pwd, is_admin) VALUES ('min', 'min_db', false);
INSERT INTO Users(user_name, pwd, is_admin) VALUES ('seo', 'seo_db', true);


INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'pea_item1', 1, 'pea', 1.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'pea_item2', 2, 'pea', 5.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'pea_item3', 3, 'pea', 10.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'pea_item4', 4, 'pea', 25.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (2, 'pea_item5', 1, 'pea', 50.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (3, 'pea_item6', 1, 'pea', 100.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (4, 'pea_item7', 1, 'pea', 500.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (5, 'pea_item8', 1, 'pea', 1000.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (6, 'pea_item9', 1, 'pea', 0.50, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'min_item1', 1, 'min', 1.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'min_item2', 2, 'min', 5.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'min_item3', 3, 'min', 10.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (1, 'min_item4', 4, 'min', 25.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (2, 'min_item5', 1, 'min', 50.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (3, 'min_item6', 1, 'min', 100.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (4, 'min_item7', 1, 'pea', 500.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (5, 'min_item8', 1, 'min', 1000.00, NOW(), '2025-05-15 00:00:00');
INSERT INTO Items(category, description, condition, seller_id, buy_it_now_price, date_posted, bid_closing_date) VALUES (6, 'min_item9', 1, 'min', 0.50, NOW(), '2025-05-15 00:00:00');


INSERT INTO Bids(item_id, bid_price, bidder_id, date_posted, bid_closing_date) SELECT i.item_id, 10.00, 'seo', NOW(), i.buy_it_now_price FROM Items as i WHERE i.description = 'pea_item7';
INSERT INTO Bids(item_id, bid_price, bidder_id, date_posted, bid_closing_date) SELECT i.item_id, 10.00, 'seo', NOW(), i.buy_it_now_price FROM Items as i WHERE i.description = 'pea_item8';
INSERT INTO Bids(item_id, bid_price, bidder_id, date_posted, bid_closing_date) SELECT i.item_id, 10.00, 'seo', NOW(), i.buy_it_now_price FROM Items as i WHERE i.description = 'pea_item9';


INSERT INTO OldBids(item_id, bid_price, bidder_id, date_posted, bid_closing_date) SELECT i.item_id, 5.00, 'min', NOW(), i.buy_it_now_price FROM Items as i WHERE i.description = 'pea_item7';
INSERT INTO OldBids(item_id, bid_price, bidder_id, date_posted, bid_closing_date) SELECT i.item_id, 5.00, 'min', NOW(), i.buy_it_now_price FROM Items as i WHERE i.description = 'pea_item8';
INSERT INTO OldBids(item_id, bid_price, bidder_id, date_posted, bid_closing_date) SELECT i.item_id, 5.00, 'min', NOW(), i.buy_it_now_price FROM Items as i WHERE i.description = 'pea_item9';

INSERT INTO Billing(item_id, sold_date, seller_id, buyer_id, price) SELECT i.item_id, NOW(), i.seller_id, 'pea', 20.00 FROM Items as i WHERE i.description = 'min_item7';
INSERT INTO Billing(item_id, sold_date, seller_id, buyer_id, price) SELECT i.item_id, NOW(), i.seller_id, 'pea', 20.00 FROM Items as i WHERE i.description = 'min_item8';
INSERT INTO Billing(item_id, sold_date, seller_id, buyer_id, price) SELECT i.item_id, NOW(), i.seller_id, 'pea', 20.00 FROM Items as i WHERE i.description = 'min_item9';
