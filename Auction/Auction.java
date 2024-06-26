import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.ResultSet;
import java.sql.DriverManager;
import java.sql.Timestamp;
import java.sql. *;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.text. *;
import java.util. *;
import java.math. *;


class UpdateDB implements Runnable {
	private static Connection connection;
	private GlobalData global_data;
	private String DB, pwd;

	public UpdateDB(GlobalData global_data, String DB, String pwd) {
		this.global_data = global_data;
		this.DB = DB;
		this.pwd = pwd;
	};

    public void run() {
        try{
            connection = DriverManager.getConnection("jdbc:postgresql://localhost/"+DB, DB, pwd);
		}
		catch(SQLException e){
			System.out.println("SQLException : " + e);	
			System.exit(1);
		}

		Queue<Integer> item_ids = new LinkedList<>();
		String find_query = "SELECT item_id FROM Items WHERE bid_closing_date < NOW()";
		String delete_query = "DELET FROM Bids WHERE item_id=?";
		String insert_query = "INSERT INTO Billing (item_id, sold_date, seller_id, buyer_id, price) " +
							  "SELECT b.item_id, NOW(), i.seller_id, b.bidder_id, b.bid_price " +
							  "FROM Bids as b JOIN Items as i ON b.item_id = i.item_id WHERE item_id = ?";

		while(!global_data.getAuctionOver()) {
			try {
				PreparedStatement pstmt = connection.prepareStatement(find_query);

				try {
					ResultSet rs = pstmt.executeQuery();
					while(rs.next()) {
						item_ids.offer(rs.getInt("item_id"));
					}
					rs.close();
				} catch (SQLException e) {
					System.out.println("Error: Invalid input is entered. Please select again.");
					continue;
				}
				pstmt.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				continue ;
			}

			while(!item_ids.isEmpty()) {
				int item_id = item_ids.poll();
				try {
					PreparedStatement pstmt = connection.prepareStatement(insert_query);
					pstmt.setInt(1, item_id);
					int rowsAffected = pstmt.executeUpdate();
					if (rowsAffected == 0){;}
					pstmt.close();

					pstmt = connection.prepareStatement(delete_query);
					pstmt.setInt(1, item_id);
					rowsAffected = pstmt.executeUpdate();
					if (rowsAffected == 0){;}
					pstmt.close();
				} catch (SQLException e) {
					System.out.println("Error: Invalid input is entered. Please select again.");
					continue ;
				}
			}
		}

		try {
			connection.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		global_data.setAuctionOver(true);
	}
}


class GlobalData {
    private boolean is_auction_over=false;
	private boolean is_update_over=false;

    public synchronized boolean getUpdateOver() {
        return this.is_update_over;
    }

	public synchronized boolean getAuctionOver() {
        return this.is_auction_over;
    }

    public synchronized void setUpdateOver(boolean data) {
        this.is_auction_over = data;
    }

    public synchronized void setAuctionOver(boolean data) {
        this.is_auction_over = data;
    }
}


public class Auction {
	private static Scanner scanner = new Scanner(System.in);
	private static String username;
	private static Connection connection;
	private static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

	public enum Category {
		ELECTRONICS(1),
		BOOKS(2),
		HOME(3),
		CLOTHING(4),
		SPORTINGGOODS(5),
		OTHERS(6);
		
		private final int value;
		
		private Category(int value) {
			this.value = value;
		}
		
		public int getValue() {
			return value;
		}
		
		public static Category getCategory(int value) {
			for (Category category : Category.values()) {
				if (category.getValue() == value) {
					return category;
				}
			}
			return null;
		}
	}
	enum Condition {
		NEW(1),
		LIKE_NEW(2),
		GOOD(3),
		ACCEPTABLE(4);
		
		private final int value;
		
		private Condition(int value) {
			this.value = value;
		}
		
		public int getValue() {
			return value;
		}
	}

	private static boolean LoginMenu() {
		String userpass, isAdmin;

		System.out.print("----< User Login >\n" +
				" ** To go back, enter 'back' in user ID.\n" +
				"     user ID: ");
		try{
			username = scanner.next();
			scanner.nextLine();

			if(username.equalsIgnoreCase("back")){
				return false;
			}

			System.out.print("     password: ");
			userpass = scanner.next();
			scanner.nextLine();
		}catch (java.util.InputMismatchException e) {
			System.out.println("Error: Invalid input is entered. Try again.");
			username = null;
			return false;
		}

		boolean login_success = false;
		String query_login = "SELECT pwd FROM Users WHERE (user_id=?) AND (pwd=?)";

		try {
			PreparedStatement pstmt = connection.prepareStatement(query_login);
			pstmt.setString(1, username);
			pstmt.setString(2, userpass);
			try {
				ResultSet rs = pstmt.executeQuery();
				if (rs.next()) {
					login_success = true;
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}
		
		if (login_success) {  
			System.out.println("You are successfully logged in.\n");
		}
		else {
			System.out.println("Error: Incorrect user name or password");
		}		
		return login_success;
	}

	private static boolean SellMenu() {
		Category category = Category.ELECTRONICS;
		Condition condition = Condition.NEW;
		char choice;
		int price;
		boolean flag_catg = true, flag_cond = true;

		do{
			System.out.println(
					"----< Sell Item >\n" +
					"---- Choose a category.\n" +
					"    1. Electronics\n" +
					"    2. Books\n" +
					"    3. Home\n" +
					"    4. Clothing\n" +
					"    5. Sporting Goods\n" +
					"    6. Other Categories\n" +
					"    P. Go Back to Previous Menu"
					);

			try {
				choice = scanner.next().charAt(0);;
			}catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				continue;
			}

			flag_catg = true;

			switch ((int) choice){
				case '1':
					category = Category.ELECTRONICS;
					continue;
				case '2':
					category = Category.BOOKS;
					continue;
				case '3':
					category = Category.HOME;
					continue;
				case '4':
					category = Category.CLOTHING;
					continue;
				case '5':
					category = Category.SPORTINGGOODS;
					continue;
				case '6':
					category = Category.OTHERS;
					continue;
				case 'p':
				case 'P':
					return false;
				default:
					System.out.println("Error: Invalid input is entered. Try again.");
					flag_catg = false;
					continue;
			}
		}while(!flag_catg);

		do{
			System.out.println(
					"---- Select the condition of the item to sell.\n" +
					"   1. New\n" +
					"   2. Like-new\n" +
					"   3. Used (Good)\n" +
					"   4. Used (Acceptable)\n" +
					"   P. Go Back to Previous Menu"
					);

			try {
				choice = scanner.next().charAt(0);;
				scanner.nextLine();
			}catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				continue;
			}

			flag_cond = true;

			switch (choice) {
				case '1':
					condition = Condition.NEW;
					break;
				case '2':
					condition = Condition.LIKE_NEW;
					break;
				case '3':
					condition = Condition.GOOD;
					break;
				case '4':
					condition = Condition.ACCEPTABLE;
					break;
				case 'p':
				case 'P':
					return false;
				default:
					System.out.println("Error: Invalid input is entered. Try again.");
					flag_cond = false;
					continue;
			}
		}while(!flag_cond);

		String desc;
		BigDecimal set_price;
		LocalDateTime dateTime;

		try {
			System.out.println("---- desc of the item (one line): ");
			desc = scanner.nextLine();
			System.out.println("---- Buy-It-Now price: ");

			while (!scanner.hasNextInt()) {
				scanner.next();
				System.out.println("Invalid input is entered. Please enter Buy-It-Now price: ");
			}

			price = scanner.nextInt();
			scanner.nextLine();
			set_price = new BigDecimal(price);

			System.out.print("---- Bid closing date and time (YYYY-MM-DD HH:MM): ");
			// you may assume users always enter valid date/time
			String date = scanner.nextLine();  /* "2023-03-04 11:30"; */
			dateTime = LocalDateTime.parse(date, formatter);
		}catch (Exception e) {
			System.out.println("Error: Invalid input is entered. Going back to the previous menu.");
			return false;
		}

		boolean is_success = false;
		String query = "INSERT INTO Items (category, desc, condition, seller_id, buy_it_now_price, bid_closing_date) VALUES (?, ?, ?, ?, ?, ?, ?)";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			
			pstmt.setInt(1, category.getValue());
			pstmt.setString(2, desc);
			pstmt.setInt(3, condition.getValue());
			pstmt.setString(4, username);
			pstmt.setBigDecimal(5, set_price);
			Timestamp timestamp = Timestamp.valueOf(dateTime);
			pstmt.setTimestamp(6, timestamp);

			int rowsAffected = pstmt.executeUpdate();
			if (rowsAffected != 0) {
				System.out.println("Your item has been successfully listed.\n");
				is_success = true;
			} else {
				System.out.println("Your item has not been successfully listed.\n");
			}
			
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}

		return is_success;
	}

	private static boolean SignupMenu() {
		/* 2. Sign Up */
		String new_username, userpass, isAdmin;
		System.out.print("----< Sign Up >\n" + 
				" ** To go back, enter 'back' in user ID.\n" +
				"---- user name: ");
		try {
			new_username = scanner.next();
			scanner.nextLine();
			if(new_username.equalsIgnoreCase("back")){
				return false;
			}
			System.out.print("---- password: ");
			userpass = scanner.next();
			scanner.nextLine();
			System.out.print("---- In this user an administrator? (Y/N): ");
			isAdmin = scanner.next();
			scanner.nextLine();
		} catch (java.util.InputMismatchException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}

		boolean signup_success = false;
		boolean is_unique_name = false;
		String query_check_name = "SELECT user_id FROM Users WHERE user_id=?";
		String query_signup = "INSERT INTO Users(user_id, pwd, is_admin) VALUES(?, ? ,?)";

		try {
			PreparedStatement pstmt = connection.prepareStatement(query_check_name);
			pstmt.setString(1, new_username);
			try {
				ResultSet rs = pstmt.executeQuery();
				if (!rs.next()) {
					is_unique_name = true;
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}

		try {
			PreparedStatement pstmt = connection.prepareStatement(query_signup);
			
			pstmt.setString(1, new_username);
			pstmt.setString(2, userpass);
			if (is_unique_name) {
				if (isAdmin.equals("Y")) {
					pstmt.setBoolean(3, true);
				}
				else {
					pstmt.setBoolean(3, false);
				}
			}
			try {
				ResultSet rs = pstmt.executeQuery();
				if (!rs.next()) {
					is_unique_name = true;
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}

		if (signup_success) {
			System.out.println("Your account has been successfully created.\n");
		}
		else {
			System.out.println("Error: Invalid input is entered. Please select again.\n");
		}
		return signup_success;
	}

	private static boolean AdminMenu() {
		/* 3. Login as Administrator */
		char choice;
		String adminname, adminpass;
		String keyword, seller;
		System.out.print("----< Login as Administrator >\n" +
				" ** To go back, enter 'back' in user ID.\n" +
				"---- admin ID: ");

		try {
			adminname = scanner.next();
			scanner.nextLine();
			if(adminname.equalsIgnoreCase("back")){
				return false;
			}
			System.out.print("---- password: ");
			adminpass = scanner.nextLine();
		} catch (java.util.InputMismatchException e) {
			System.out.println("Error: Invalid input is entered. Try again.");
			return false;
		}

		boolean login_success = true;
		String query_login = "SELECT pwd FROM Users WHERE (user_id=?) and (pwd=?) and (is_admin=true)";

		try {
			PreparedStatement pstmt = connection.prepareStatement(query_login);
			pstmt.setString(1, adminname);
			pstmt.setString(2, adminpass);
			try {
				ResultSet rs = pstmt.executeQuery();
				if (rs.next()) {
					login_success = true;
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}

		if(!login_success){
			System.out.println("Error: Incorrect user name or password or you are not admin");
			return false;
		}

		do {
			System.out.println(
					"----< Admin menu > \n" +
					"    1. Print Sold Items per Category \n" +
					"    2. Print Account Balance for Seller \n" +
					"    3. Print Seller Ranking \n" +
					"    4. Print Buyer Ranking \n" +
					"    P. Go Back to Previous Menu"
					);

			try {
				choice = scanner.next().charAt(0);;
				scanner.nextLine();
			} catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				continue;
			}

			if (choice == '1') {
				System.out.println("----Enter Category to search : ");
				keyword = scanner.next();
				scanner.nextLine();
				System.out.println("sold item       | sold date       | seller ID   | buyer ID   | price | commissions");
				System.out.println("----------------------------------------------------------------------------------");

				String query = "SELECT item_id, sold_date, seller_id, buyer_id, price FROM Billing as billing WHERE billing.item_id IN SELECT item_id FROM Items WHERE category=?";
				try {
					PreparedStatement pstmt = connection.prepareStatement(query);
					pstmt.setString(1, keyword);
					String seller_id, buyer_id;
					BigDecimal price, commissions;
					Timestamp sold_date;
					int item_id;
					 
					try {
						ResultSet rs = pstmt.executeQuery();
						while(rs.next()) {
							item_id = rs.getInt("item_id");
							seller_id = rs.getString("seller_id");
							buyer_id = rs.getString("buyer_id");
							price = rs.getBigDecimal("price");
							commissions = price.divide(new BigDecimal("10"));
							sold_date = rs.getTimestamp("sold_date");
							System.out.println(String.valueOf(item_id)+"\t"+sold_date.toLocalDateTime().format(formatter)+"\t"+seller_id+"\t"+buyer_id+"\t"+String.valueOf(price)+"\t"+String.valueOf(commissions));
						} 
						rs.close();
					} catch (SQLException e) {
						System.out.println("Error: Invalid input is entered. Please select again.");
						return false;
					}
					pstmt.close();
				}
				catch (SQLException e) {
					System.out.println("Error: Invalid input is entered. Please select again.");
					return false;
				}
				continue;

			} else if (choice == '2') {
				/*TODO: Print Account Balance for Seller */
				System.out.println("---- Enter Seller ID to search : ");
				seller = scanner.next();
				scanner.nextLine();
				System.out.println("sold item       | sold date       | buyer ID   | price | commissions");
				System.out.println("--------------------------------------------------------------------");

				String query = "SELECT item_id, sold_date, buyer_id, price FROM Billing as billing WHERE seller_id=?";
				try {
					PreparedStatement pstmt = connection.prepareStatement(query);
					pstmt.setString(1, seller);
					String buyer_id;
					BigDecimal price, commissions;
					Timestamp sold_date;
					int item_id;
					 
					try {
						ResultSet rs = pstmt.executeQuery();
						while(rs.next()) {
							item_id = rs.getInt("item_id");
							buyer_id = rs.getString("buyer_id");
							price = rs.getBigDecimal("price");
							commissions = price.divide(new BigDecimal("10"));
							sold_date = rs.getTimestamp("sold_date");
							System.out.println(String.valueOf(item_id)+"\t"+sold_date.toLocalDateTime().format(formatter)+"\t"+buyer_id+"\t"+String.valueOf(price)+"\t"+String.valueOf(commissions));
						}
						rs.close();
					} catch (SQLException e) {
						System.out.println("Error: Invalid input is entered. Please select again.");
						return false;
					}
					pstmt.close();
				} catch (SQLException e) {
					System.out.println("Error: Invalid input is entered. Please select again.");
					return false;
				}
				continue;

			} else if (choice == '3') {
				/*TODO: Print Seller Ranking */
				System.out.println("seller ID   | # of items sold | Total Profit (excluding commissions)");
				System.out.println("--------------------------------------------------------------------");

				String query = "SELECT seller_id, COUNT(item_id) as num_item, SUM(price) as total_price FROM Billing GROUP BY seller_id ORDER BY total_pricedesc";
				try {
					PreparedStatement pstmt = connection.prepareStatement(query);
					String seller_id;
					int num_item;
					BigDecimal total_price;
					 
					try {
						ResultSet rs = pstmt.executeQuery();
						while(rs.next()) {
							seller_id = rs.getString("seller_id");
							total_price = rs.getBigDecimal("total_price");
							total_price = total_price.multiply(BigDecimal.valueOf(0.9));
							num_item = rs.getInt("num_item");
							System.out.println(seller_id+"\t"+String.valueOf(num_item)+"\t"+String.valueOf(total_price));
						}
						rs.close();
					} catch (SQLException e) {
						System.out.println("Error: Invalid input is entered. Please select again.");
						return false;
					}
					pstmt.close();
				} catch (SQLException e) {
					System.out.println("Error: Invalid input is entered. Please select again.");
					return false;
				}

				continue;
			} else if (choice == '4') {
				/*TODO: Print Buyer Ranking */
				System.out.println("buyer ID   | # of items purchased | Total Money Spent ");
				System.out.println("------------------------------------------------------");
				
				String query = "SELECT buyer_id, COUNT(item_id) as num_item, SUM(price) as total_price FROM Billing GROUP BY buyer_id ORDER BY total_pricedesc";
				try {
					PreparedStatement pstmt = connection.prepareStatement(query);
					String buyer_id;
					int num_item;
					BigDecimal total_price;
					 
					try {
						ResultSet rs = pstmt.executeQuery();
						while(rs.next()) {
							buyer_id = rs.getString("buyer_id");
							total_price = rs.getBigDecimal("total_price");
							num_item = rs.getInt("num_item");
							System.out.println(buyer_id+"\t"+String.valueOf(num_item)+"\t"+String.valueOf(total_price));
						}
						rs.close();
					} catch (SQLException e) {
						System.out.println("Error: Invalid input is entered. Please select again.");
						return false;
					}
					pstmt.close();
				} catch (SQLException e) {
					System.out.println("Error: Invalid input is entered. Please select again.");
					return false;
				}
				
				continue;
			} else if (choice == 'P' || choice == 'p') {
				return false;
			} else {
				System.out.println("Error: Invalid input is entered. Try again.");
				continue;
			}
		} while(true);
	}

	public static void CheckSellStatus(){
		System.out.println("item listed in Auction | bidder (buyer ID) | bidding price | bidding date/time \n");
		System.out.println("-------------------------------------------------------------------------------\n");

		String query = "SELECT b.item_id, b.bidder_id, b.bid_price, b.date_posted FROM Bids as b JOIN Items as i ON b.item_id = i.item_id WHERE i.seller_id=?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String bidder_id;
			Timestamp bid_date;
			BigDecimal bid_price;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					bidder_id = rs.getString("bidder_id");
					bid_price = rs.getBigDecimal("bid_price");
					bid_date = rs.getTimestamp("bid_date");
					System.out.println(String.valueOf(item_id)+"\t"+bidder_id+"\t"+String.valueOf(bid_price)+"\t"+bid_date.toLocalDateTime().format(formatter));
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return;
		}

	}

	public static boolean BuyItem(){
		Category category = Category.ELECTRONICS;
		Condition condition = Condition.NEW;
		char choice;
		int price;
		String keyword, seller, datePosted;
		boolean flag_catg = true, flag_cond = true;
		
		do {

			System.out.println( "----< Select category > : \n" +
					"    1. Electronics\n"+
					"    2. Books\n" + 
					"    3. Home\n" + 
					"    4. Clothing\n" + 
					"    5. Sporting Goods\n" +
					"    6. Other categories\n" +
					"    7. Any category\n" +
					"    P. Go Back to Previous Menu"
					);

			try {
				choice = scanner.next().charAt(0);;
				scanner.nextLine();
			} catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				return false;
			}

			flag_catg = true;

			switch (choice) {
				case '1':
					category = Category.ELECTRONICS;
					break;
				case '2':
					category = Category.BOOKS;
					break;
				case '3':
					category = Category.HOME;
					break;
				case '4':
					category = Category.CLOTHING;
					break;
				case '5':
					category = Category.SPORTINGGOODS;
					break;
				case '6':
					category = Category.OTHERS;
					break;
				case '7':
					break;
				case 'p':
				case 'P':
					return false;
				default:
					System.out.println("Error: Invalid input is entered. Try again.");
					flag_catg = false;
					continue;
			}
		} while(!flag_catg);

		do {

			System.out.println(
					"----< Select the condition > \n" +
					"   1. New\n" +
					"   2. Like-new\n" +
					"   3. Used (Good)\n" +
					"   4. Used (Acceptable)\n" +
					"   P. Go Back to Previous Menu"
					);
			try {
				choice = scanner.next().charAt(0);;
				scanner.nextLine();
			} catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				return false;
			}

			flag_cond = true;

			switch (choice) {
				case '1':
					condition = Condition.NEW;
					break;
				case '2':
					condition = Condition.LIKE_NEW;
					break;
				case '3':
					condition = Condition.GOOD;
					break;
				case '4':
					condition = Condition.ACCEPTABLE;
					break;
				case 'p':
				case 'P':
					return false;
				default:
					System.out.println("Error: Invalid input is entered. Try again.");
					flag_cond = false;
					continue;
				}
		} while(!flag_cond);

		try {
			System.out.println("---- Enter keyword to search the desc : ");
			keyword = scanner.next();
			scanner.nextLine();

			System.out.println("---- Enter Seller ID to search : ");
			System.out.println(" ** Enter 'any' if you want to see items from any seller. ");
			seller = scanner.next();
			scanner.nextLine();

			System.out.println("---- Enter date posted (YYYY-MM-DD): ");
			System.out.println(" ** This will search items that have been posted after the designated date.");
			System.out.println(" ** Enter 'any' if you want to see items for all dates. ");
			datePosted = scanner.next();
			scanner.nextLine();
		} catch (java.util.InputMismatchException e) {
			System.out.println("Error: Invalid input is entered. Try again.");
			return false;
		}

		System.out.println("Item ID | Item desc | Condition | Seller | Buy-It-Now | Current Bid | highest bidder | Time left | bid close");
		System.out.println("-------------------------------------------------------------------------------------------------------");
		
		String query = "SELECT i.item_id as item_id, i.desc as desc, i.condition as condition, i.seller_id as seller_id, i.buy_it_now_price as buy_it_now_price, b.bid_price as bid_price, b.bidder_id as bidder_id, (NOW() - i.bid_closing_date) as time_left, i.bid_closing_date as bid_closing_date FROM Items as i LEFT JOIN Bids as b ON i.item_id = b.item_id WHERE i.category=? AND i.condition=? AND i.desc ILIKE ?";
		
		if (!(seller.equals("any"))) {
			query = query.concat(" AND i.seller_id=?");
		}
		if (!(datePosted.equals("any"))) {
			query = query.concat(" AND i.date_posted>=?");
		}
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String desc, seller_id, bidder_id;
			Timestamp bid_closing_date, time_left;
			BigDecimal bid_price, buy_it_now_price;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					desc = rs.getString("desc");
					seller_id = rs.getString("seller_id");
					bidder_id = rs.getString("bidder_id");
					buy_it_now_price = rs.getBigDecimal("buy_it_now_price");
					bid_price = rs.getBigDecimal("bid_price");
					time_left = rs.getTimestamp("time_left");
					bid_closing_date = rs.getTimestamp("bid_closing_date");
					System.out.println(String.valueOf(item_id)+"\t"+desc+"\t"+condition+"\t"+seller_id+"\t"+String.valueOf(buy_it_now_price)+"\t"+String.valueOf(bid_price)+"\t"+bidder_id+"\t"+time_left.toLocalDateTime().format(formatter)+"\t"+bid_closing_date.toLocalDateTime().format(formatter));
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}

		System.out.println("---- Select Item ID to buy or bid: ");

		String item_id_str;
		try {
			item_id_str = scanner.nextLine();
			scanner.nextLine();
			System.out.println("     Price: ");
			price = scanner.nextInt();
			scanner.nextLine();
		} catch (java.util.InputMismatchException e) {
			System.out.println("Error: Invalid input is entered. Try again.");
			return false;
		}

		int comparisonResult, item_id = Integer.parseInt(item_id_str);
		BigDecimal buy_it_now_price, bid_price;
		buy_it_now_price = BigDecimal.ZERO;
		bid_price = BigDecimal.ZERO;
		boolean right_choice = false;
		String find_query = "SELECT i.buy_it_now_price as buy_it_now_price, b.bid_price as bid_price FROM Items as i LEFT JOIN Bids as b ON i.item_id=b.item_id WHERE item_id = ?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(find_query);
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					buy_it_now_price = rs.getBigDecimal("buy_it_now_price");
					bid_price = rs.getBigDecimal("bid_price");
					right_choice = true;
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return false;
		}
		if (!right_choice) {
			System.out.println("Invalid choice is entered. Try again.");
			return false;
		}

		BigDecimal new_price = BigDecimal.valueOf(price);

		if (bid_price == null) {
			String insert_new_query = "INSERT INTO Bids (item_id, bid_price, bidder_id, date_posted) " +
									  "SELECT ?, ?, ?, NOW() ";
			
			try {
				PreparedStatement pstmt = connection.prepareStatement(insert_new_query);
				pstmt.setInt(1, item_id);
				pstmt.setBigDecimal(2, new_price);
				pstmt.setString(3, username);
				int rowsAffected = pstmt.executeUpdate();
				if (rowsAffected == 0){;}
				pstmt.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}

			System.out.println("Congratulations, the item is yours now.\n");
		}

		comparisonResult = bid_price.compareTo(new_price);
		if (comparisonResult <= 0) {
			System.out.println("Invalid price is entered. Try again.");
			return false;
		}

		comparisonResult = buy_it_now_price.compareTo(new_price);
		if (comparisonResult <= 0) {
			String delete_query = "DELET FROM Bids WHERE item_id=?";
			String insert_query = "INSERT INTO Billing (item_id, sold_date, seller_id, buyer_id, price) " +
								"SELECT b.item_id, NOW(), i.seller_id, b.bidder_id, b.bid_price " +
								"FROM Bids as b JOIN Items as i ON b.item_id = i.item_id WHERE item_id = ?";
			
			try {
				PreparedStatement pstmt = connection.prepareStatement(insert_query);
				pstmt.setInt(1, item_id);
				int rowsAffected = pstmt.executeUpdate();
				if (rowsAffected == 0){;}
				pstmt.close();

				pstmt = connection.prepareStatement(delete_query);
				pstmt.setInt(1, item_id);
				rowsAffected = pstmt.executeUpdate();
				if (rowsAffected == 0){;}
				pstmt.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}

			System.out.println("Congratulations, the item is yours now.\n");
		} else {
			String delete_query = "DELET FROM Bids WHERE item_id=?";
			String insert_old_query = "INSERT INTO OldBids (item_id, bid_price, bidder_id, date_posted) " +
									  "SELECT b.item_id, b.bid_price, b.bidder_id, b.date_posted " +
									  "FROM Bids as b JOIN Items as i ON b.item_id = i.item_id WHERE item_id = ?";
			String insert_new_query = "INSERT INTO Bids (item_id, bid_price, bidder_id, date_posted) " +
									  "SELECT ?, ?, ?, NOW() ";
			
			try {
				PreparedStatement pstmt = connection.prepareStatement(insert_old_query);
				pstmt.setInt(1, item_id);
				int rowsAffected = pstmt.executeUpdate();
				if (rowsAffected == 0){;}
				pstmt.close();

				pstmt = connection.prepareStatement(delete_query);
				pstmt.setInt(1, item_id);
				rowsAffected = pstmt.executeUpdate();
				if (rowsAffected == 0){;}
				pstmt.close();

				pstmt = connection.prepareStatement(delete_query);
				pstmt.setInt(1, item_id);
				pstmt.setBigDecimal(2, new_price);
				pstmt.setString(3, username);
				rowsAffected = pstmt.executeUpdate();
				if (rowsAffected == 0){;}
				pstmt.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return false;
			}

			System.out.println("Congratulations, you are the highest bidder.\n"); 
		}
		
		return true;
	}

	public static void CheckBuyStatus(){
		/* TODO: Check the status of the item the current buyer is bidding on */
		/* Even if you are outbidded or the bid closing date has passed, all the items this user has bidded on must be displayed */

		System.out.println("item ID   | item desc   | highest bidder | highest bidding price | your bidding price | bid closing date/time");
		System.out.println("--------------------------------------------------------------------------------------------------------------------");
		
		String query = "SELECT b.item_id, i.decription, b.bid_price i.bid_closing_date FROM Bids as b JOIN Items as i ON b.item_id = i.item_id WHERE b.bidder_id=?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String desc;
			Timestamp bid_closing_date;
			BigDecimal bid_price;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					desc = rs.getString("desc");
					bid_price = rs.getBigDecimal("bid_price");
					bid_closing_date = rs.getTimestamp("bid_closing_date");
					System.out.println(String.valueOf(item_id)+"\t"+desc+"\t"+username+"\t"+String.valueOf(bid_price)+"\t"+String.valueOf(bid_price)+"\t"+bid_closing_date.toLocalDateTime().format(formatter));
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return ;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return ;
		}

		query = "SELECT o.item_id as item_id,  i.desc as dscription, b.bidder_id as highest_bidder, b.bid_price as highest_price, o.bid_price as bid_price i.bid_closing_date as bid_closing_date FROM OldBids as o LEFT JOIN Bids as b ON o.item_id=b.item_id LEFT JOIN Items as i ON o.item_id=i.item_id WHERE b.bidder_id=?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String desc, highest_bidder;
			Timestamp bid_closing_date;
			BigDecimal bid_price, highest_price;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					desc = rs.getString("desc");
					highest_bidder = rs.getString("highest_bidder");
					bid_price = rs.getBigDecimal("bid_price");
					highest_price = rs.getBigDecimal("highest_price");
					bid_closing_date = rs.getTimestamp("bid_closing_date");
					System.out.println(String.valueOf(item_id)+"\t"+desc+"\t"+highest_bidder+"\t"+String.valueOf(highest_price)+"\t"+String.valueOf(bid_price)+"\t"+bid_closing_date.toLocalDateTime().format(formatter));
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return;
		}

		query = "SELECT o.item_id as item_id,  i.desc as dscription, b.buyer_id as highest_bidder, b.price as highest_price, o.bid_price as bid_price i.bid_closing_date as bid_closing_date FROM OldBids as o LEFT JOIN Billing as b ON o.item_id=b.item_id LEFT JOIN Items as i ON o.item_id=i.item_id WHERE b.bidder_id=?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String desc, highest_bidder;
			Timestamp bid_closing_date;
			BigDecimal bid_price, highest_price;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					desc = rs.getString("desc");
					highest_bidder = rs.getString("highest_bidder");
					bid_price = rs.getBigDecimal("bid_price");
					highest_price = rs.getBigDecimal("highest_price");
					bid_closing_date = rs.getTimestamp("bid_closing_date");
					System.out.println(String.valueOf(item_id)+"\t"+desc+"\t"+highest_bidder+"\t"+String.valueOf(highest_price)+"\t"+String.valueOf(bid_price)+"\t"+bid_closing_date.toLocalDateTime().format(formatter));
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return;
		}
	}

	public static void CheckAccount(){
		/* TODO: Check the balance of the current user.  */
		System.out.println("[Sold Items] \n");
		System.out.println("item category  | item ID   | sold date | sold price  | buyer ID | commission  ");
		System.out.println("------------------------------------------------------------------------------");
		
		String query = "SELECT i.category, b.item_id, b.sold_date, b.price, b.buyer_id FROM Billing as b JOIN Items as i ON b.item_id = i.item_id WHERE b.seller_id=?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String buyer_id;
			Category category;
			Timestamp sold_date;
			BigDecimal price, commissions;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					category = Category.getCategory(rs.getInt("category"));
					buyer_id = rs.getString("buyer_id");
					price = rs.getBigDecimal("price");
					commissions = price.divide(new BigDecimal("10"));
					sold_date = rs.getTimestamp("sold_date");
					System.out.println(category+"\t"+String.valueOf(item_id)+"\t"+sold_date.toLocalDateTime().format(formatter)+"\t"+String.valueOf(price)+"\t"+buyer_id+"\t"+String.valueOf(commissions));
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return;
		}
		
		System.out.println("[Purchased Items] \n");
		System.out.println("item category  | item ID   | purchased date | puchased price  | seller ID ");
		System.out.println("--------------------------------------------------------------------------");
		query = "SELECT i.category, b.item_id, b.sold_date, b.price, b.seller_id FROM Billing as b JOIN Items as i ON b.item_id = i.item_id WHERE b.buyer_id=?";
		try {
			PreparedStatement pstmt = connection.prepareStatement(query);
			pstmt.setString(1, username);
			String seller_id;
			Category category;
			Timestamp sold_date;
			BigDecimal price;
			int item_id;
				
			try {
				ResultSet rs = pstmt.executeQuery();
				while(rs.next()) {
					item_id = rs.getInt("item_id");
					category = Category.getCategory(rs.getInt("category"));
					seller_id = rs.getString("seller_id");
					price = rs.getBigDecimal("price");
					sold_date = rs.getTimestamp("sold_date");
					System.out.println(category+"\t"+String.valueOf(item_id)+"\t"+sold_date.toLocalDateTime().format(formatter)+"\t"+String.valueOf(price)+"\t"+seller_id+"\t");
				}
				rs.close();
			} catch (SQLException e) {
				System.out.println("Error: Invalid input is entered. Please select again.");
				return;
			}
			pstmt.close();
		} catch (SQLException e) {
			System.out.println("Error: Invalid input is entered. Please select again.");
			return;
		}
	}

	public static void main(String[] args) {
		char choice;
		boolean ret;
		GlobalData global_data = new GlobalData();

		if(args.length<2){
			System.out.println("Usage: java Auction postgres_id password");
			System.exit(1);
		}


		try{
            //    	conn = DriverManager.getConnection("jdbc:postgresql://localhost/"+args[0], args[0], args[1]); 
            connection = DriverManager.getConnection("jdbc:postgresql://localhost/"+args[0], args[0], args[1]);
		}
		catch(SQLException e){
			System.out.println("SQLException : " + e);	
			System.exit(1);
		}

		global_data.setAuctionOver(false);
		global_data.setUpdateOver(false);
		Thread workerThread = new Thread(new UpdateDB(global_data, args[0], args[1]));
        workerThread.start();

		do {
			username = null;
			System.out.println(
					"----< Login menu >\n" + 
					"----(1) Login\n" +
					"----(2) Sign up\n" +
					"----(3) Login as Administrator\n" +
					"----(Q) Quit"
					);

			try {
				choice = scanner.next().charAt(0);;
				scanner.nextLine();
			} catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				continue;
			}

			try {
				switch ((int) choice) {
					case '1':
						ret = LoginMenu();
						if(!ret) continue;
						break;
					case '2':
						ret = SignupMenu();
						if(!ret) continue;
						break;
					case '3':
						ret = AdminMenu();
						if(!ret) continue;
					case 'q':
					case 'Q':
						System.out.println("Good Bye");
						/* TODO: close the connection and clean up everything here */
						connection.close();
						global_data.setAuctionOver(true);
						while(!global_data.getUpdateOver()) {;}
						System.exit(1);
					default:
						System.out.println("Error: Invalid input is entered. Try again.");
				}
			} catch (SQLException e) {
				System.out.println("SQLException : " + e);	
			}
		} while (username==null || username.equalsIgnoreCase("back"));  

		// logged in as a normal user 
		do {
			System.out.println(
					"---< Main menu > :\n" +
					"----(1) Sell Item\n" +
					"----(2) Status of Your Item Listed on Auction\n" +
					"----(3) Buy Item\n" +
					"----(4) Check Status of your Bid \n" +
					"----(5) Check your Account \n" +
					"----(Q) Quit"
					);

			try {
				choice = scanner.next().charAt(0);;
				scanner.nextLine();
			} catch (java.util.InputMismatchException e) {
				System.out.println("Error: Invalid input is entered. Try again.");
				continue;
			}

			try{
				switch (choice) {
					case '1':
						ret = SellMenu();
						if(!ret) continue;
						break;
					case '2':
						CheckSellStatus();
						break;
					case '3':
						ret = BuyItem();
						if(!ret) continue;
						break;
					case '4':
						CheckBuyStatus();
						break;
					case '5':
						CheckAccount();
						break;
					case 'q':
					case 'Q':
						System.out.println("Good Bye");
						connection.close();
						global_data.setAuctionOver(true);
						while(!global_data.getUpdateOver()) {;}
						System.exit(1);
				}
			} catch (SQLException e) {
				System.out.println("SQLException : " + e);	
				try {
					connection.close();
				} catch (SQLException es) {
					es.printStackTrace();
				}
				global_data.setAuctionOver(true);
				while(!global_data.getUpdateOver()) {;}
				System.exit(1);
			}
		} while(true);
	} // End of main 
} // End of class


